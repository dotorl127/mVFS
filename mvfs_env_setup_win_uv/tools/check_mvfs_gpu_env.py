# -*- coding: utf-8 -*-
from __future__ import annotations

import torch

print("torch=", torch.__version__)
print("cuda=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
print("gpu=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

try:
    import diffusers, transformers, accelerate, cv2, lpips
    print("diffusers=", diffusers.__version__)
    print("transformers=", transformers.__version__)
    print("accelerate=", accelerate.__version__)
    print("opencv=", cv2.__version__)
    print("lpips=OK")
except Exception as e:
    print("[WARN] package check failed:", e)
