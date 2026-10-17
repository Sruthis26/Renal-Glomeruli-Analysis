import os
import sys
import uuid
import shutil
import asyncio
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import subprocess
from run_pipeline import resolve_qupath_exe

app = FastAPI(title="Renal Glomeruli API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "output"
STATIC_DIR = PROJECT_ROOT / "static"

for d in [UPLOADS_DIR, OUTPUT_DIR, STATIC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

runs = {}

def get_dataset_pairs(directory: Path) -> List[dict]:
    pairs = []
    if not directory.exists():
        return pairs
    
    svs_files = list(directory.glob("*.svs"))
    for svs in svs_files:
        json_file = directory / f"{svs.stem}.svs.json"
        if json_file.exists():
            pairs.append({
                "id": f"{directory.name}/{svs.stem}",
                "name": svs.name,
                "svs_path": str(svs),
                "json_path": str(json_file),
                "source": directory.name
            })
    return pairs

@app.get("/api/datasets")
def list_datasets():
    datasets = []
    datasets.extend(get_dataset_pairs(DATASET_DIR))
    
    # Check uploads directory (organized by uuid folders)
    for upload_folder in UPLOADS_DIR.iterdir():
        if upload_folder.is_dir():
            datasets.extend(get_dataset_pairs(upload_folder))
            
    return {"datasets": datasets}

@app.post("/api/upload")
async def upload_dataset(
    svs_file: UploadFile = File(...),
    json_file: UploadFile = File(...)
):
    if not svs_file.filename.endswith(".svs"):
        raise HTTPException(status_code=400, detail="svs_file must be an .svs file")
    if not json_file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="json_file must be a .json file")
        
    upload_id = str(uuid.uuid4())
    target_dir = UPLOADS_DIR / upload_id
    target_dir.mkdir(parents=True, exist_ok=True)
    
    svs_path = target_dir / svs_file.filename
    json_path = target_dir / f"{Path(svs_file.filename).stem}.svs.json"
    
    with svs_path.open("wb") as f:
        shutil.copyfileobj(svs_file.file, f)
        
    with json_path.open("wb") as f:
        shutil.copyfileobj(json_file.file, f)
        
    return {
        "message": "Upload successful",
        "dataset": {
            "id": f"{upload_id}/{svs_path.stem}",
            "name": svs_path.name,
            "svs_path": str(svs_path),
            "json_path": str(json_path),
            "source": upload_id
        }
    }

def run_pipeline_task(run_id: str, svs_path: str, json_path: str, no_qupath: bool):
    runs[run_id] = {"status": "running", "output": "", "error": ""}
    
    cmd = [
        sys.executable, "run_pipeline.py",
        "--svs", svs_path,
        "--json", json_path
    ]
    if no_qupath:
        cmd.append("--no-qupath")
        
    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            cwd=str(PROJECT_ROOT),
            text=True
        )
        
        if process.returncode == 0:
            runs[run_id]["status"] = "completed"
            runs[run_id]["output"] = process.stdout
            
            # Find the output folder for this slide
            stem = Path(svs_path).stem
            # if name is like foo.svs, stem is foo. Wait, Path('foo.svs').stem -> foo
            # run_pipeline.py creates output in output/{stem}
            case_output_dir = OUTPUT_DIR / stem
            
            results = {}
            if case_output_dir.exists():
                for file_path in case_output_dir.glob("*"):
                    if file_path.is_file():
                        # Map internal path to url path
                        # e.g., output/grundium17/grundium17_aif_overlay.png -> /output/grundium17/...
                        url_path = f"/output/{stem}/{file_path.name}"
                        if file_path.suffix == ".png":
                            if "aif_overlay" in file_path.name:
                                results["aif_overlay"] = url_path
                            elif "artery_overlay" in file_path.name:
                                results["artery_overlay"] = url_path
                        elif file_path.suffix == ".pdf":
                            results["pdf_summary"] = url_path
                        elif file_path.suffix == ".json":
                            results["json_summary"] = url_path
                        elif file_path.suffix == ".csv":
                            if "results_csv" not in results:
                                results["results_csv"] = []
                            results["results_csv"].append(url_path)
                            
            runs[run_id]["results"] = results
        else:
            runs[run_id]["status"] = "failed"
            runs[run_id]["error"] = f"Exit code {process.returncode}.\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
            
    except Exception as e:
        runs[run_id]["status"] = "failed"
        runs[run_id]["error"] = f"{type(e).__name__}: {str(e)}"

@app.post("/api/run")
async def start_run(
    svs_path: str = Form(...),
    json_path: str = Form(...),
    no_qupath: bool = Form(False),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    run_id = str(uuid.uuid4())
    background_tasks.add_task(run_pipeline_task, run_id, svs_path, json_path, no_qupath)
    return {"run_id": run_id, "status": "started"}

@app.get("/api/status/{run_id}")
def get_status(run_id: str):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run ID not found")
    return runs[run_id]

@app.post("/api/open-qupath")
def open_qupath(svs_path: str = Form(...)):
    stem = Path(svs_path).stem
    import re
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    project_file = PROJECT_ROOT / "qupath_projects" / safe_stem / "project.qpproj"
    
    if not project_file.exists():
        raise HTTPException(status_code=404, detail="QuPath project not found. Make sure you ran the pipeline without the 'no-qupath' flag.")
        
    try:
        qupath_exe = resolve_qupath_exe()
        # Launch detached process so it doesn't block the server
        subprocess.Popen([qupath_exe, "--project", str(project_file)])
        return {"message": "QuPath launched successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return JSONResponse({"message": "Please navigate to /static/index.html to view the app."})
