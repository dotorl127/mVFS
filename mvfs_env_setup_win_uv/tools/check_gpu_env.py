import sys

def main() -> int:
    print("python =", sys.version.replace("\n", " "))

    try:
        import torch
        print("torch =", torch.__version__)
        print("torch cuda =", torch.version.cuda)
        print("torch cuda available =", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("gpu =", torch.cuda.get_device_name(0))
            x = torch.randn(1, device="cuda")
            print("torch cuda tensor ok =", x.item() is not None)
    except Exception as e:
        print("[ERROR] torch check failed:", repr(e))
        return 1

    try:
        import onnxruntime as ort
        # ORT CUDA provider can often use PyTorch's CUDA/cuDNN DLLs when torch is imported first.
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
                print("onnxruntime preload_dlls ok")
            except Exception as e:
                print("[WARN] onnxruntime preload_dlls failed:", repr(e))
        print("onnxruntime =", ort.__version__)
        print("onnxruntime providers =", ort.get_available_providers())
    except Exception as e:
        print("[ERROR] onnxruntime check failed:", repr(e))
        return 1

    try:
        import cv2, numpy, PIL
        print("opencv =", cv2.__version__)
        print("numpy =", numpy.__version__)
        print("PIL =", PIL.__version__)
    except Exception as e:
        print("[WARN] image stack check failed:", repr(e))

    try:
        import insightface
        print("insightface =", getattr(insightface, "__version__", "unknown"))
    except Exception as e:
        print("[WARN] insightface import failed:", repr(e))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
