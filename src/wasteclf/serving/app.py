"""FastAPI inference service.

Endpoints
---------
``GET  /health``         liveness and the loaded label set
``GET  /classes``        the class list in model index order
``POST /predict``        one image, ``multipart/form-data``
``POST /predict/batch``  several images in one request

The model is loaded once at application startup rather than per request, and a
warm-up pass runs before the service reports ready, so the first real request
does not pay the graph tracing cost.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np

from wasteclf import __version__
from wasteclf.inference.predictor import Predictor
from wasteclf.utils.logging import get_logger

# Imported at module scope rather than inside create_app(). Route annotations
# are strings under `from __future__ import annotations`, and pydantic resolves
# them against this module's globals; a name bound only inside a function body
# is not there, and FastAPI then fails every request with "not fully defined".
# wasteclf.serving.__init__ defers importing this module, so a user without the
# serve extra never reaches this import.
try:
    from fastapi import FastAPI, File, HTTPException, UploadFile

    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    FASTAPI_AVAILABLE = False

logger = get_logger(__name__)

#: Refuse uploads larger than this. Without a cap, one request can exhaust the
#: worker's memory before decoding ever gets a chance to fail.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

#: Cap on images per batch request, for the same reason.
MAX_BATCH_SIZE = 64


def create_app(run_dir: str | Path, threshold: float = 0.0):
    """Build the FastAPI application for one trained run.

    Args:
        run_dir: A finished run directory holding ``model.keras`` and
            ``labels.json``.
        threshold: Predictions below this confidence are flagged
            ``low_confidence``. The label is still returned; the caller decides.
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("serving requires FastAPI: pip install 'wasteclf[serve]'")

    predictor = Predictor.from_run(run_dir, threshold=threshold)

    @asynccontextmanager
    async def lifespan(app):  # noqa: ARG001 - FastAPI passes the app; unused here
        dummy = np.zeros((1, *predictor.image_size, 3), dtype=np.float32)
        started = time.perf_counter()
        predictor.model.predict(dummy, verbose=0)
        logger.info("warm-up pass took %.0f ms", (time.perf_counter() - started) * 1000)
        yield

    app = FastAPI(
        title="Waste classifier",
        version=__version__,
        description="Image classification over the waste taxonomy this run was trained on.",
        lifespan=lifespan,
    )

    async def read_upload(upload: UploadFile) -> bytes:
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"{upload.filename}: empty upload")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"{upload.filename}: {len(data)} bytes exceeds "
                    f"the {MAX_UPLOAD_BYTES} byte limit"
                ),
            )
        return data

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "classes": predictor.class_names,
            "image_size": list(predictor.image_size),
        }

    @app.get("/classes")
    async def classes() -> dict:
        return {"classes": predictor.class_names, "count": len(predictor.class_names)}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> dict:
        data = await read_upload(file)
        started = time.perf_counter()
        try:
            result = predictor.predict_bytes(data, name=file.filename or "upload")
        except Exception as exc:  # noqa: BLE001 - a decode failure is the client's error
            raise HTTPException(
                status_code=400, detail=f"could not decode {file.filename!r}: {exc}"
            ) from exc
        payload = result.to_dict()
        payload["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return payload

    @app.post("/predict/batch")
    async def predict_batch(files: list[UploadFile] = File(...)) -> dict:
        if len(files) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=413, detail=f"at most {MAX_BATCH_SIZE} images per request"
            )

        images: list[np.ndarray] = []
        names: list[str] = []
        failures: list[dict] = []

        for upload in files:
            data = await read_upload(upload)
            name = upload.filename or "upload"
            try:
                images.append(predictor.load_bytes(data))
                names.append(name)
            except Exception as exc:  # noqa: BLE001
                # One undecodable file must not fail the whole batch.
                failures.append({"path": name, "error": str(exc)})

        if not images:
            raise HTTPException(status_code=400, detail="no decodable images in the request")

        started = time.perf_counter()
        results = predictor.predict_array(np.stack(images), names)
        return {
            "predictions": [r.to_dict() for r in results],
            "failed": failures,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    return app
