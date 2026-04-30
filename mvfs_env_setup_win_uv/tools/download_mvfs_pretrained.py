from __future__ import annotations

import argparse
from pathlib import Path


def download_sd_turbo(root: Path) -> None:
    from huggingface_hub import snapshot_download

    out = root / "sd-turbo"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[SD-Turbo] downloading to {out}")
    snapshot_download(
        repo_id="stabilityai/sd-turbo",
        local_dir=str(out),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("[SD-Turbo] done")


def download_insightface_buffalo_l() -> None:
    print("[InsightFace] initializing buffalo_l; model pack will be auto-downloaded if missing.")

    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
        except Exception as e:
            print("[WARN] ort.preload_dlls failed:", repr(e))

    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("[InsightFace] buffalo_l ready")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-root", default="pretrained", type=str)
    p.add_argument("--sd-turbo", action="store_true")
    p.add_argument("--insightface-buffalo-l", action="store_true")
    args = p.parse_args()

    root = Path(args.pretrained_root)
    root.mkdir(parents=True, exist_ok=True)

    if args.sd_turbo:
        download_sd_turbo(root)
    if args.insightface_buffalo_l:
        download_insightface_buffalo_l()

    if not args.sd_turbo and not args.insightface_buffalo_l:
        print("Nothing selected. Use --sd-turbo and/or --insightface-buffalo-l")


if __name__ == "__main__":
    main()
