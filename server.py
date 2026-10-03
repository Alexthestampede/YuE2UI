#!/usr/bin/env python3
"""YuE2 webui: small FastAPI server with a one-job-at-a-time GPU worker."""
from __future__ import annotations

import json
import os
import random
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
SONGS_DIR = Path(os.environ.get("YUE2UI_SONGS_DIR", ROOT / "outputs" / "webui"))
SONGS_DIR.mkdir(parents=True, exist_ok=True)
MODEL = os.environ.get("YUE2UI_MODEL", "m-a-p/YuE2-3B")
VAE = os.environ.get("YUE2UI_VAE", "m-a-p/YuE2-Vae")

# ROCm/MIOpen: the default FIND_MODE runs a per-shape kernel search (~40-90s
# per new tensor shape) that is not persisted across processes. FAST picks a
# known-good kernel immediately; identical output, near-zero first-call cost.
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

app = FastAPI(title="YuE2 webui")

_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_queue: list[str] = []
_worker_ready = threading.Event()


def _set(job_id, **kw):
    with _lock:
        _jobs[job_id].update(kw)


class GenerateRequest(BaseModel):
    style: str
    lyrics: str
    cot: str = "full"
    seed: int | None = None
    guidance: float = 1.0
    abc: str | None = None
    # Reroll/preview workflow (SqualusShiraii pattern): reuse a saved plan and
    # skip the ~3-25 min AR score planning; only semantic+NAR+VAE run.
    plan_dir: str | None = None
    # Draft mode: 16 NAR ODE steps (house default per listening tests);
    # full quality = 32. Preview and full are the SAME take, different render.
    preview: bool = True


def _worker():
    import torch
    from yue2 import YuE2Pipeline
    from yue2.pipeline import SongRequest, SongResult
    from yue2.storage import identity

    pipe = None
    while True:
        job_id = None
        with _lock:
            if _queue:
                job_id = _queue.pop(0)
        if job_id is None:
            time.sleep(0.5)
            continue
        job = _jobs[job_id]
        req = GenerateRequest(**job["request"])
        out_dir = SONGS_DIR / job_id
        _set(job_id, status="running", stage="loading model", started=time.time())
        try:
            if pipe is None:
                # torch-eager is ~2.1x faster than CUDA-graph execution on
                # ROCm (measured RX 7800 XT: semantic 7.0 -> 22.3 tok/s);
                # NVIDIA keeps the graph path. Override via YUE2UI_BACKEND.
                import torch as _torch
                backend = os.environ.get("YUE2UI_BACKEND") or (
                    "torch-eager" if _torch.version.hip else "torch")
                pipe = YuE2Pipeline.from_pretrained(MODEL, vae=VAE, device="cuda",
                                                    backend=backend)
            if req.preview:
                from dataclasses import replace as _dc_replace
                from yue2.protocol import GenerationConfig
                pipe.generation_config = _dc_replace(pipe.generation_config, ode_steps=16)
            else:
                from dataclasses import replace as _dc_replace
                from yue2.protocol import GenerationConfig
                pipe.generation_config = _dc_replace(pipe.generation_config, ode_steps=32)
            plan_dir = req.plan_dir
            if plan_dir and not Path(plan_dir).is_absolute():
                candidate = SONGS_DIR / plan_dir
                plan_dir = candidate if (candidate / "plan_manifest.json").is_file() else Path(plan_dir)
            if plan_dir:
                from yue2.pipeline import SymbolicPlan
                plan = SymbolicPlan.load(Path(plan_dir))
                request = plan.request
                _set(job_id, stage="reusing saved plan")
            else:
                request = SongRequest(style=req.style, lyrics=req.lyrics, cot=req.cot,
                                      seed=req.seed if req.seed is not None else random.randrange(2**31),
                                      cfg_scale=req.guidance,
                                      abc=req.abc if req.abc and req.cot in ("full", "melody") else None)
            config = pipe.effective_config(request)
            request_id = identity({"request": request.to_dict(), "config": config, "weights": pipe.weights})
            start = time.time()
            tokens = {"abc": 0, "semantic": 0}

            def on_token(phase, _token):
                tokens[phase] = tokens.get(phase, 0) + 1
                _set(job_id, tokens=tokens["abc"] + tokens["semantic"])

            _set(job_id, stage="reusing saved plan" if plan_dir else "planning score")
            if not plan_dir:
                plan = pipe.plan(request=request, on_token=on_token)
            else:
                _set(job_id, tokens=0)
            _set(job_id, stage="generating song", abc=plan.abc)
            semantic = pipe.generate_semantic(plan, on_token=on_token)
            _set(job_id, stage="synthesizing audio")
            t_nar = time.time()
            latents = pipe.synthesize(semantic)
            nar_seconds = time.time() - t_nar
            _set(job_id, stage="decoding audio")
            t_vae = time.time()
            audio = pipe.decode(latents)
            vae_seconds = time.time() - t_vae
            timing = {"abc": plan.timing, "semantic": semantic.timing, "nar_seconds": nar_seconds,
                      "vae_seconds": vae_seconds, "e2e_seconds": time.time() - start}
            song = SongResult(audio, 48000, semantic, latents, config, pipe.weights, timing, request_id)
            result = song.save_artifacts(out_dir)
            _set(job_id, status="done", stage="done", finished=time.time(),
                 result={"audio": f"/api/jobs/{job_id}/audio", "abc": f"/api/jobs/{job_id}/abc",
                         "seconds": result["audio_seconds"], "timing": timing,
                         "truncated": result["truncated"], "dir": str(out_dir)})
        except Exception as e:  # noqa: BLE001
            import traceback
            _set(job_id, status="error", stage="error", finished=time.time(),
                 error=f"{type(e).__name__}: {e}", trace=traceback.format_exc())


threading.Thread(target=_worker, daemon=True).start()


@app.post("/api/generate")
def generate(req: GenerateRequest):
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {"id": job_id, "status": "queued", "stage": "queued",
                         "request": req.model_dump(), "tokens": 0,
                         "created": time.time()}
        _queue.append(job_id)
    return {"id": job_id}


@app.get("/api/jobs/{job_id}/abc/raw")
def job_abc_raw(job_id: str):
    path = SONGS_DIR / job_id / "score.abc"
    if not path.exists():
        raise HTTPException(404)
    from fastapi import Response
    return Response(path.read_text(encoding="utf-8"), media_type="text/plain")


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404)
    return {k: v for k, v in job.items() if k not in {"request", "trace"}}


@app.get("/api/jobs/{job_id}/audio")
def job_audio(job_id: str):
    path = SONGS_DIR / job_id / "audio.flac"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/flac", filename=f"yue2-{job_id}.flac")


@app.get("/api/jobs/{job_id}/abc")
def job_abc(job_id: str):
    path = SONGS_DIR / job_id / "score.abc"
    if not path.exists():
        raise HTTPException(404)
    return HTMLResponse(f"<pre>{path.read_text(encoding='utf-8')}</pre>")


@app.get("/api/jobs/{job_id}/trace")
def job_trace(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404)
    return HTMLResponse(f"<pre>{job.get('trace', 'no trace')}</pre>")


@app.get("/api/songs")
def songs():
    out = []
    for d in sorted(SONGS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        rf = d / "result.json"
        if not rf.is_file():
            continue
        r = json.loads(rf.read_text())
        jid = d.name
        out.append({"id": jid, "seconds": r.get("audio_seconds"),
                    "style": (d / "request.json") and json.loads((d / "request.json").read_text()).get("style", "")[:80],
                    "audio": f"/api/jobs/{jid}/audio",
                    "e2e": (r.get("timing") or {}).get("e2e_seconds")})
    return JSONResponse(out)


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("YUE2UI_HOST", "127.0.0.1"),
                port=int(os.environ.get("YUE2UI_PORT", "7860")))