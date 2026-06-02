"""
Environment diagnostic script.

Run after `pip install -r requirements.txt` to confirm:
  * Python version is 3.11.x
  * All required libraries import cleanly
  * PyTorch sees the GPU (CUDA) if one is available
  * The DATASET folder is present and the class counts look right

Usage (from the project root, with the venv activated):
    python -m src.check_env
"""
from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"

def ok(msg: str)   -> None: print(f"{GREEN}[ OK ]{RESET} {msg}")
def warn(msg: str) -> None: print(f"{YELLOW}[WARN]{RESET} {msg}")
def fail(msg: str) -> None: print(f"{RED}[FAIL]{RESET} {msg}")


# --------------------------------------------------------------------- #
# 1. Python version                                                     #
# --------------------------------------------------------------------- #
def check_python() -> None:
    print("\n--- Python ---")
    v = sys.version_info
    print(f"      Interpreter: {sys.executable}")
    print(f"      Version    : {platform.python_version()}")
    if (v.major, v.minor) == (3, 11):
        ok("Python 3.11 detected (target version).")
    elif (v.major, v.minor) in [(3, 10), (3, 12)]:
        warn(f"Python {v.major}.{v.minor} works but 3.11 is the tested target.")
    else:
        fail(f"Python {v.major}.{v.minor} is NOT supported. Recreate the venv with `py -3.11 -m venv .venv`.")


# --------------------------------------------------------------------- #
# 2. Library imports                                                    #
# --------------------------------------------------------------------- #
REQUIRED = [
    ("torch",         None),
    ("torchvision",   None),
    ("timm",          None),
    ("numpy",         None),
    ("pandas",        None),
    ("cv2",           "opencv-python"),
    ("PIL",           "Pillow"),
    ("sklearn",       "scikit-learn"),
    ("matplotlib",    None),
    ("seaborn",       None),
    ("tqdm",          None),
    ("tensorboard",   None),
    ("pytorch_grad_cam", "grad-cam"),
    ("streamlit",     None),
    ("jupyter",       None),
]

def check_imports() -> None:
    print("\n--- Required libraries ---")
    for module_name, pip_name in REQUIRED:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "?")
            ok(f"{module_name:20s} {version}")
        except Exception as e:                            # noqa: BLE001
            display = pip_name or module_name
            fail(f"{module_name:20s} import failed -> pip install {display}  ({e})")


# --------------------------------------------------------------------- #
# 3. GPU / CUDA                                                         #
# --------------------------------------------------------------------- #
def check_gpu() -> None:
    print("\n--- GPU / CUDA ---")
    try:
        import torch
    except Exception:
        fail("torch is not importable; skipping GPU check.")
        return

    print(f"      torch    : {torch.__version__}")
    print(f"      CUDA wheel: {torch.version.cuda}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        ok(f"CUDA available — {n} device(s) detected.")
        for i in range(n):
            name = torch.cuda.get_device_name(i)
            cap  = torch.cuda.get_device_capability(i)
            mem  = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"      [{i}] {name}  (compute {cap[0]}.{cap[1]}, {mem:.1f} GB VRAM)")
        # quick GPU smoke test
        try:
            x = torch.randn(1024, 1024, device="cuda")
            y = (x @ x).sum().item()
            ok(f"GPU matmul smoke test passed (result={y:.2f}).")
        except Exception as e:                            # noqa: BLE001
            fail(f"GPU matmul failed: {e}")
    else:
        warn("CUDA NOT available — training will run on CPU (~6x slower).")


# --------------------------------------------------------------------- #
# 4. Dataset sanity                                                     #
# --------------------------------------------------------------------- #
def check_dataset() -> None:
    print("\n--- Dataset ---")
    try:
        from src.config import TRAIN_DIR, TEST_DIR, CLASS_NAMES
    except Exception as e:                                # noqa: BLE001
        fail(f"Could not import src.config ({e}).")
        return

    for split_name, split_dir in [("Training", TRAIN_DIR), ("Testing", TEST_DIR)]:
        if not split_dir.is_dir():
            fail(f"Missing folder: {split_dir}")
            continue
        ok(f"Found {split_name} at {split_dir}")
        for cls in CLASS_NAMES:
            cls_dir = split_dir / cls
            if not cls_dir.is_dir():
                fail(f"  Missing class folder: {cls_dir}")
                continue
            n = sum(1 for p in cls_dir.iterdir() if p.is_file())
            ok(f"  {cls:12s} {n:>5d} images")


# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #
def main() -> int:
    print("=" * 60)
    print(" Brain Tumor Detection — Environment Diagnostic ")
    print("=" * 60)
    # Make `src` importable when run as `python src/check_env.py` too.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    check_python()
    check_imports()
    check_gpu()
    check_dataset()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
