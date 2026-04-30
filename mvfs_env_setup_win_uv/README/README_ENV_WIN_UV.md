# MVFS Windows 11 + uv + CUDA 12.x 환경 세팅

## 권장 환경 분리

- `.venv-mvfs`: Python 3.10, SD-Turbo / InsightFace / 학습용
- `.venv-3ddfa`: Python 3.8, 3DDFA-V3 전처리용

3DDFA-V3는 별도 환경으로 두는 편이 의존성 충돌이 훨씬 적다.

## 순서

```bat
bat\00_install_uv_win.bat
bat\01_setup_mvfs_env_win_cuda12.bat
bat\02_download_mvfs_pretrained.bat
bat\03_setup_3ddfa_env_win_cuda12_py38.bat
bat\04_download_3ddfa_assets.bat
```

## CUDA stack

- MVFS main env:
  - Python 3.10
  - PyTorch 2.4.1 + CUDA 12.1 wheel
  - ONNX Runtime GPU 1.20.1
  - InsightFace buffalo_l

- 3DDFA env:
  - Python 3.8
  - PyTorch 2.1.2 + CUDA 12.1 wheel

PyTorch pip wheel은 자체 CUDA runtime을 포함하므로 로컬 CUDA Toolkit 설치가 필수는 아니지만,
NVIDIA driver는 CUDA 12.x wheel을 실행할 수 있을 만큼 최신이어야 한다.

## pretrained

- `pretrained/sd-turbo`: Hugging Face `stabilityai/sd-turbo`
- InsightFace `buffalo_l`: `~/.insightface/models/buffalo_l`에 자동 다운로드
- 3DDFA-V3 assets: `3ddfa-v3/assets`

## 주의

- InsightFace model pack은 사용 조건/라이선스를 반드시 확인해야 한다.
- 3DDFA-V3 assets는 Hugging Face dataset `Zidu-Wang/3DDFA-V3`에서 받는다.
- Windows에서 ONNX Runtime GPU가 CUDA provider를 못 잡으면 `tools/check_gpu_env.py` 결과를 먼저 확인한다.
