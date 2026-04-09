import os
import io
import uuid
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import demucs.api

app = FastAPI(title="Demucs Audio Separator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("/home/saikiran/demucs-api/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load separator once at startup
separator = None


def get_separator():
    global separator
    if separator is None:
        separator = demucs.api.Separator(model="htdemucs", segment=10)
    return separator


@app.on_event("startup")
async def startup():
    get_separator()


@app.get("/health")
async def health():
    return {"status": "ok", "model": "htdemucs"}


@app.post("/separate")
async def separate_audio(file: UploadFile = File(...)):
    """
    Upload an audio file -> get back vocals and no_vocals (accompaniment) as downloadable files.
    Returns a job_id with paths to the separated files.
    """
    job_id = str(uuid.uuid4())
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True)

    # Save uploaded file
    input_path = job_dir / f"input_{file.filename}"
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        sep = get_separator()
        origin, separated = sep.separate_audio_file(str(input_path))

        vocals_path = job_dir / "vocals.wav"
        no_vocals_path = job_dir / "no_vocals.wav"

        # Save vocals
        vocals = separated["vocals"]
        demucs.api.save_audio(vocals, str(vocals_path), samplerate=sep.samplerate)

        # Combine all non-vocal stems into accompaniment
        non_vocal_keys = [k for k in separated.keys() if k != "vocals"]
        accompaniment = sum(separated[k] for k in non_vocal_keys)
        demucs.api.save_audio(accompaniment, str(no_vocals_path), samplerate=sep.samplerate)

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
    This streams the result directly without saving to disk.
    """
    content = await file.read()

    try:
        sep = get_separator()

        # Write to temp file for demucs
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        origin, separated = sep.separate_audio_file(tmp_path)
        os.unlink(tmp_path)

        if stem == "vocals":
            audio = separated["vocals"]
        else:
            non_vocal_keys = [k for k in separated.keys() if k != "vocals"]
            audio = sum(separated[k] for k in non_vocal_keys)

        # Encode to WAV in memory
        buf = io.BytesIO()
        torchaudio.save(buf, audio, sep.samplerate, format="wav")
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename={stem}.wav"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8686)
