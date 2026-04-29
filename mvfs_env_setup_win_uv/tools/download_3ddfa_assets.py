from __future__ import annotations

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="Zidu-Wang/3DDFA-V3")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[3DDFA-V3] Downloading assets from {args.repo_id}")
    print(f"[3DDFA-V3] Output: {out.resolve()}")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(out),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print("[DONE] 3DDFA-V3 assets downloaded.")
    print("Check whether files are nested under assets/assets. If so, move them one level up.")


if __name__ == "__main__":
    main()
