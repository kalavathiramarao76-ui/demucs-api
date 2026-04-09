import os
import io
import uuid
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import torch
import soundfile as sf
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile

app = FastAPI(title="Demucs Audio Separator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("/home/saikiran/demucs-api/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

model = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_model_instance():
    global model
    if model is None:
        model = get_model("htdemucs")
        model.to(DEVICE)
        model.eval()
    return model


def save_audio(tensor, path_or_buf, samplerate):
    """Save a torch tensor as WAV using soundfile. Tensor shape: (channels, samples)."""
    audio_np = tensor.numpy().T  # soundfile expects (samples, channels)
    sf.write(path_or_buf, audio_np, samplerate, format="WAV", subtype="FLOAT")


def separate_file(input_path: str):
    """Run demucs separation on an audio file. Returns dict of stem_name -> tensor."""
    mdl = get_model_instance()
    wav = AudioFile(input_path).read(streams=0, samplerate=mdl.samplerate, channels=mdl.audio_channels)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    wav = wav.to(DEVICE)

    with torch.no_grad():
        sources = apply_model(mdl, wav[None], split=True, overlap=0.25, progress=False)[0]

    sources = sources * ref.std() + ref.mean()

    result = {}
    for i, name in enumerate(mdl.sources):
        result[name] = sources[i].cpu()

    return result, mdl.samplerate


STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
async def homepage():
    return (STATIC_DIR / "index.html").read_text()


@app.on_event("startup")
async def startup():
    get_model_instance()


@app.get("/health")
async def health():
    return {"status": "ok", "model": "htdemucs", "device": DEVICE}


@app.post("/separate")
async def separate_audio(file: UploadFile = File(...)):
    """
    Upload an audio file -> get back vocals and no_vocals (accompaniment) as downloadable files.
    Returns a job_id with paths to the separated files.
    """
    job_id = str(uuid.uuid4())
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True)

    input_path = job_dir / f"input_{file.filename}"
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        stems, sr = separate_file(str(input_path))

        vocals_path = job_dir / "vocals.wav"
        no_vocals_path = job_dir / "no_vocals.wav"

        # Save vocals
        save_audio(stems["vocals"], str(vocals_path), sr)

        # Combine all non-vocal stems into accompaniment
        non_vocal_keys = [k for k in stems if k != "vocals"]
        accompaniment = sum(stems[k] for k in non_vocal_keys)
        save_audio(accompaniment, str(no_vocals_path), sr)

        # Clean up input file
        input_path.unlink(missing_ok=True)

        return JSONResponse({
            "job_id": job_id,
            "vocals": f"/download/{job_id}/vocals.wav",
            "no_vocals": f"/download/{job_id}/no_vocals.wav",
        })
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/separate/stream")
async def separate_audio_stream(
    file: UploadFile = File(...),
    stem: str = Query("vocals", enum=["vocals", "no_vocals"]),
):
    """
    Upload an audio file -> stream back the selected stem (vocals or no_vocals) as WAV.
    """
    content = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        stems, sr = separate_file(tmp_path)
        os.unlink(tmp_path)

        if stem == "vocals":
            audio = stems["vocals"]
        else:
            non_vocal_keys = [k for k in stems if k != "vocals"]
            audio = sum(stems[k] for k in non_vocal_keys)

        buf = io.BytesIO()
        save_audio(audio, buf, sr)
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename={stem}.wav"},
        )
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/separate/batch")
async def separate_batch(files: list[UploadFile] = File(..., description="Multiple audio files to separate")):
    """
    Upload multiple audio files for batch separation.
    Each file gets vocals and no_vocals extracted.
    Returns a job_id with per-file results and a ZIP download link.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    job_id = str(uuid.uuid4())
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True)

    results = {}
    failed = []

    for upload in files:
        fname = upload.filename or f"input_{uuid.uuid4().hex[:6]}.wav"
        safe_name = os.path.splitext(fname)[0]
        input_path = job_dir / f"input_{fname}"

        with open(input_path, "wb") as f:
            content = await upload.read()
            f.write(content)

        try:
            stems, sr = separate_file(str(input_path))

            vocals_name = f"{safe_name}_vocals.wav"
            no_vocals_name = f"{safe_name}_no_vocals.wav"

            save_audio(stems["vocals"], str(job_dir / vocals_name), sr)

            non_vocal_keys = [k for k in stems if k != "vocals"]
            accompaniment = sum(stems[k] for k in non_vocal_keys)
            save_audio(accompaniment, str(job_dir / no_vocals_name), sr)

            results[fname] = {
                "status": "completed",
                "vocals": f"/download/{job_id}/{vocals_name}",
                "no_vocals": f"/download/{job_id}/{no_vocals_name}",
            }
        except Exception as e:
            results[fname] = {"status": "failed", "error": str(e)}
            failed.append(fname)
        finally:
            input_path.unlink(missing_ok=True)

    return JSONResponse({
        "job_id": job_id,
        "total_files": len(files),
        "completed": len(files) - len(failed),
        "failed": len(failed),
        "files": results,
        "download_zip": f"/download/{job_id}/all.zip",
    })


@app.get("/download/{job_id}/all.zip")
async def download_batch_zip(job_id: str):
    """Download all separated files from a batch job as a ZIP archive."""
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    wav_files = [f for f in job_dir.iterdir() if f.is_file() and f.suffix == ".wav"]
    if not wav_files:
        raise HTTPException(status_code=404, detail="No output files found")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for wav_file in wav_files:
            zf.write(wav_file, wav_file.name)
    zip_buf.seek(0)

    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="batch_{job_id}.zip"'},
    )


@app.get("/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download a separated audio file by job_id."""
    file_path = OUTPUT_DIR / job_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path), media_type="audio/wav", filename=filename)


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Check what files are available for a given job."""
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    files = [f.name for f in job_dir.iterdir() if f.is_file()]
    return {"job_id": job_id, "files": files}


# ── Fix Swagger UI for batch file upload ─────────────────────────────────────

from fastapi.openapi.utils import get_openapi


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title, version=app.version, description=app.description, routes=app.routes,
    )
    # Fix batch file upload in both component schemas and inline schemas
    for name, comp in schema.get("components", {}).get("schemas", {}).items():
        props = comp.get("properties", {})
        if "files" in props:
            props["files"] = {
                "type": "array",
                "items": {"type": "string", "format": "binary"},
                "title": "Audio Files",
                "description": "Select multiple audio files (use Add Item to add more)",
            }
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8686)
