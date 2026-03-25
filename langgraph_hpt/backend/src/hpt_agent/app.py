from __future__ import annotations

import json
import pathlib
import sys
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

_SRC_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from hpt_agent.configuration import SystemConfig
from hpt_agent.graph import run_once, stream_run_events
from hpt_agent.plugin_loader import register_task_from_files
from hpt_agent.storage import TaskStorage

app = FastAPI(title="LangGraph HPT Backend")

_system = SystemConfig.from_env()
_storage = TaskStorage(_system.storage_root)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/hpt/tasks/register")
async def register_task(
    plugin_file: UploadFile = File(...),
    search_space_file: UploadFile = File(...),
    task_name: Optional[str] = Form(default=None),
    entrypoint: str = Form(default="objective"),
):
    try:
        plugin_tmp = _system.storage_root / f"tmp_{plugin_file.filename}"
        plugin_tmp.parent.mkdir(parents=True, exist_ok=True)
        plugin_tmp.write_bytes(await plugin_file.read())

        space_tmp = _system.storage_root / f"tmp_{search_space_file.filename}"
        space_tmp.write_bytes(await search_space_file.read())
        json.loads(space_tmp.read_text(encoding="utf-8"))

        task_id = register_task_from_files(
            storage=_storage,
            plugin_file=plugin_tmp,
            search_space_file=space_tmp,
            task_name=task_name,
            entrypoint=entrypoint,
        )
        return {"task_id": task_id}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        if "plugin_tmp" in locals() and plugin_tmp.exists():
            plugin_tmp.unlink(missing_ok=True)
        if "space_tmp" in locals() and space_tmp.exists():
            space_tmp.unlink(missing_ok=True)


@app.get("/hpt/tasks/{task_id}")
def get_task(task_id: str):
    try:
        task = _storage.load_task(task_id)
        payload = {
            "task_id": task["task_id"],
            "entrypoint": task["entrypoint"],
            "signature": task["signature"],
            "manifest": task["manifest"],
            "search_space_raw": task["search_space_raw"],
        }
        return JSONResponse(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/hpt/runs/invoke")
def run_invoke(payload: dict):
    try:
        result = run_once(payload)
        return JSONResponse(result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/hpt/runs/stream")
def run_stream(payload: dict):
    def event_gen():
        try:
            for event in stream_run_events(payload):
                event_name = event.get("phase", "update")
                yield (
                    f"event: {event_name}\n"
                    f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
                )
            yield "event: done\ndata: {}\n\n"
        except Exception as exc:
            err = {"phase": "run_failed", "error": str(exc)}
            yield f"event: run_failed\ndata: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


def create_frontend_router(build_dir="../frontend/dist"):
    build_path = pathlib.Path(__file__).parent.parent.parent / build_dir
    if not build_path.is_dir() or not (build_path / "index.html").is_file():
        from starlette.routing import Route

        async def dummy_frontend(_request):
            return JSONResponse(
                {
                    "message": "Frontend not built. Run 'npm run build' in frontend directory."
                },
                status_code=503,
            )

        return Route("/{path:path}", endpoint=dummy_frontend)
    return StaticFiles(directory=build_path, html=True)


app.mount("/app", create_frontend_router(), name="frontend")
