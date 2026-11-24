"""Download an explicitly pinned public encoder into the ignored local model folder."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
TARGET = ROOT / "project" / "models" / "bge-small-en-v1.5"
FILES = ["config.json", "model.safetensors", "modules.json", "sentence_bert_config.json",
         "tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json",
         "1_Pooling/config.json"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", action="store_true", help="Also download and upstream-verify the pinned official FP32 ONNX export")
    args = parser.parse_args()
    TARGET.mkdir(parents=True, exist_ok=True)
    manifest = {"model": "BAAI/bge-small-en-v1.5", "revision": REVISION, "files": []}
    for name in FILES:
        path = TARGET / name
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/{REVISION}/{name}"
        print(f"Downloading {name}", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "SDSC5001-research"})
        temporary = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(path)
        manifest["files"].append({"path": name, "bytes": path.stat().st_size,
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (TARGET / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Pinned model ready at {TARGET}", flush=True)
    if args.onnx:
        from probe_onnx import download_onnx
        download_onnx(TARGET)


if __name__ == "__main__":
    main()
