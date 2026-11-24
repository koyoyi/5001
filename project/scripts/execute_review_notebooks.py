"""Execute all review notebooks and record the exact run, cells and file hashes."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import asyncio
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "project/.vendor-jupyter"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))
    os.environ["PYTHONPATH"] = str(VENDOR) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--kernel", default="python3")
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    if not (artifact / "search_bundle.joblib").is_file():
        parser.error("Pass a completed local run with its search bundle.")
    for name, folder in (("IPYTHONDIR", "ipython"), ("JUPYTER_RUNTIME_DIR", "jupyter")):
        directory = ROOT / "tmp" / folder
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(directory)
    os.environ["RAG_ARTIFACT"] = str(artifact)
    os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
    records = []
    async def execute_books():
        for path in sorted((ROOT / "project/notebooks").glob("*.ipynb")):
            book = nbformat.read(path, as_version=4)
            manager = AsyncKernelManager(kernel_name=args.kernel)
            client = NotebookClient(book, km=manager, timeout=180, kernel_name=args.kernel,
                                    resources={"metadata": {"path": str(ROOT)}})
            try:
                await client.async_execute()
            finally:
                # Stop heartbeat/I/O clients before the server disappears. This avoids
                # Windows libzmq reset assertions during otherwise successful shutdown.
                if client.kc is not None:
                    client.kc.stop_channels()
                if manager.has_kernel:
                    await manager.shutdown_kernel(now=False)
            cells = [cell for cell in book.cells if cell.cell_type == "code"]
            assert all(cell.execution_count is not None for cell in cells)
            assert not any(output.output_type == "error" for cell in cells for output in cell.outputs)
            nbformat.write(book, path)
            records.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                            "sha256": sha(path), "executed_code_cells": len(cells)})
            print("Passed " + path.name, flush=True)
    asyncio.run(execute_books())
    report = {"status": "passed", "executed_at_utc": datetime.now(timezone.utc).isoformat(),
              "run_id": artifact.name, "kernel": args.kernel, "results_sha256": sha(artifact / "results.json"),
              "notebooks": records}
    (ROOT / "project/notebooks/execution_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
