"""Vercel entrypoint.

Serves an exported ONNX model. Deliberately does not import TensorFlow: the
full stack is about 1.2 GB installed and will not fit in a serverless function,
while onnxruntime, Pillow and NumPy come to roughly 180 MB and do.

The model is not in the repository. Export one into ``api/model/``::

    wasteclf export --run runs/<name> --format onnx --out api/model

Until that exists the service still starts and ``/health`` explains what is
missing, rather than crashing the function on every cold start with a
FileNotFoundError nobody can read from the Vercel dashboard.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# The package uses a src layout and is not pip-installed here: installing it
# would pull in the training dependencies, TensorFlow included, which is the
# one thing this function exists to avoid.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402

from wasteclf.inference.onnx_predictor import OnnxPredictor  # noqa: E402

MODEL_DIR = Path(os.environ.get("WASTECLF_MODEL_DIR", Path(__file__).parent / "model"))
THRESHOLD = float(os.environ.get("WASTECLF_THRESHOLD", "0.0"))

#: Serverless request bodies are capped well below this anyway; the point is to
#: reject early rather than decode something enormous.
MAX_UPLOAD_BYTES = 4 * 1024 * 1024

app = FastAPI(
    title="Waste classifier",
    description="ONNX inference over the waste taxonomy the model was trained on.",
)

_predictor: OnnxPredictor | None = None
_load_error: str | None = None


def get_predictor() -> OnnxPredictor:
    """Load the model once per warm instance."""
    global _predictor, _load_error
    if _predictor is None:
        try:
            _predictor = OnnxPredictor.from_dir(MODEL_DIR, threshold=THRESHOLD)
            _load_error = None
        except Exception as exc:  # noqa: BLE001 - reported through /health
            _load_error = f"{type(exc).__name__}: {exc}"
            raise HTTPException(
                status_code=503,
                detail=(
                    f"no model loaded ({_load_error}). Export one with: "
                    "wasteclf export --run runs/<name> --format onnx --out api/model"
                ),
            ) from exc
    return _predictor


@app.get("/health")
async def health() -> dict:
    """Liveness, plus whether a model is actually present."""
    try:
        predictor = get_predictor()
    except HTTPException as exc:
        return {"status": "no_model", "detail": exc.detail, "model_dir": str(MODEL_DIR)}
    return {
        "status": "ok",
        "classes": predictor.class_names,
        "image_size": list(predictor.image_size),
        "runtime": "onnxruntime",
    }


@app.get("/classes")
async def classes() -> dict:
    predictor = get_predictor()
    return {"classes": predictor.class_names, "count": len(predictor.class_names)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    predictor = get_predictor()

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"{file.filename}: empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{file.filename}: {len(data)} bytes exceeds the {MAX_UPLOAD_BYTES} byte limit",
        )

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


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """A small upload form, so the deployment is usable from a browser."""
    try:
        classes_note = ", ".join(get_predictor().class_names)
    except HTTPException:
        classes_note = "no model loaded — see /health"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Waste classifier</title>
<style>
  :root {{ color-scheme: light dark; --fg: #111; --bg: #fff; --muted: #666; --line: #ddd; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg: #e8e8e8; --bg: #111; --muted: #999; --line: #333; }}
  }}
  body {{ font: 16px/1.6 system-ui, sans-serif; max-width: 34rem; margin: 0 auto;
         padding: 2.5rem 1rem; color: var(--fg); background: var(--bg); }}
  h1 {{ font-size: 1.4rem; margin-bottom: .25rem; }}
  p.sub {{ color: var(--muted); margin-top: 0; }}
  form {{ border: 1px solid var(--line); border-radius: 8px; padding: 1.25rem; margin: 1.5rem 0; }}
  button {{ font: inherit; padding: .5rem 1rem; margin-top: .75rem; cursor: pointer; }}
  pre {{ background: rgba(128,128,128,.12); padding: 1rem; border-radius: 6px;
        overflow-x: auto; font-size: .85rem; }}
  code {{ font-size: .9em; }}
</style>
</head>
<body>
  <h1>Waste classifier</h1>
  <p class="sub">Classes: {classes_note}</p>

  <form id="f">
    <input type="file" name="file" accept="image/*" required>
    <button type="submit">Classify</button>
  </form>

  <pre id="out">POST an image to /predict, or use the form above.</pre>

  <script>
    const f = document.getElementById('f'), out = document.getElementById('out');
    f.addEventListener('submit', async (e) => {{
      e.preventDefault();
      out.textContent = 'classifying...';
      try {{
        const r = await fetch('/predict', {{ method: 'POST', body: new FormData(f) }});
        out.textContent = JSON.stringify(await r.json(), null, 2);
      }} catch (err) {{
        out.textContent = 'request failed: ' + err;
      }}
    }});
  </script>
</body>
</html>"""
