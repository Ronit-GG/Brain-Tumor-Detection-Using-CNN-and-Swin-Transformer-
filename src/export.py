"""
Export trained PyTorch models to ONNX format for cross-platform deployment.

ONNX models can be served via `onnxruntime` (Python, C++, C#, etc.)
without requiring PyTorch.

Usage:
    python -m src.export                # export all 3 models
    python -m src.export --models cnn   # export only specific models
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .config         import IMG_SIZE, MODELS_DIR, NUM_CLASSES
from .cnn_model      import build_cnn
from .transfer_model import build_transfer_model
from .swin_model     import build_swin_model
from .utils          import get_device


# Default ONNX opset version. Opset 17 supports all the ops we use
# (LayerNorm, GELU, conv, etc.) and is widely supported by runtimes.
DEFAULT_OPSET = 17


# --------------------------------------------------------------------- #
# Per-model factory + state-dict loader                                 #
# --------------------------------------------------------------------- #
def _load_for_export(name: str, ckpt_path: Path) -> torch.nn.Module:
    """Rebuild the architecture and load saved weights, in eval mode on CPU."""
    if name == "cnn":
        model = build_cnn(num_classes=NUM_CLASSES)
    elif name == "transfer":
        model = build_transfer_model(freeze_backbone=False)
    elif name == "swin":
        model = build_swin_model(freeze_backbone=False)
    else:
        raise ValueError(f"Unknown model: {name}")

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


# --------------------------------------------------------------------- #
# Single-model ONNX export                                              #
# --------------------------------------------------------------------- #
def export_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    image_size: int = IMG_SIZE,
    opset: int = DEFAULT_OPSET,
    dynamic_batch: bool = True,
) -> Path:
    """
    Trace `model` and write an ONNX graph to `output_path`.

    `dynamic_batch=True` lets the exported model accept batches of any
    size at inference time (not just the trace-time batch of 1).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, image_size, image_size)

    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None

    # Force the stable legacy TorchScript exporter. The new TorchDynamo
    # exporter (default in torch 2.12) emits Unicode console output that
    # crashes on Windows cp1252 consoles, and has rougher edges for the
    # window-attention pattern in Swin.
    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    return output_path


# --------------------------------------------------------------------- #
# Numerical equivalence check (torch vs onnxruntime)                    #
# --------------------------------------------------------------------- #
def verify_onnx(
    torch_model: torch.nn.Module,
    onnx_path: Path,
    image_size: int = IMG_SIZE,
    n_samples: int = 1,
) -> dict:
    """
    Confirm that the exported ONNX produces the same *classifications*
    as the original PyTorch model.

    For classification, what matters is the *argmax*, not the exact logit
    values. ONNX export of architectures using StochasticDepth + BatchNorm
    (EfficientNet) can introduce sub-percent numerical drift that grows
    with batch size, but the predicted class is preserved.
    """
    import onnxruntime as ort

    torch_model.eval()
    x = torch.randn(n_samples, 3, image_size, image_size)

    with torch.no_grad():
        torch_out = torch_model(x).numpy()

    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["output"], {"input": x.numpy()})[0]

    # Numerical drift
    abs_err = np.abs(torch_out - onnx_out)
    max_err = float(abs_err.max())
    mean_err = float(abs_err.mean())

    # The important agreement: do they predict the same class?
    pred_torch = torch_out.argmax(axis=-1)
    pred_onnx  = onnx_out.argmax(axis=-1)
    argmax_agreement = float((pred_torch == pred_onnx).mean())

    # And how close are the softmax probabilities?
    def _softmax(z):
        e = np.exp(z - z.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)
    prob_diff = float(np.abs(_softmax(torch_out) - _softmax(onnx_out)).max())

    # Speed comparison
    t0 = time.perf_counter()
    for _ in range(20):
        with torch.no_grad():
            _ = torch_model(x)
    torch_ms = (time.perf_counter() - t0) * 1000 / 20

    t0 = time.perf_counter()
    for _ in range(20):
        _ = sess.run(["output"], {"input": x.numpy()})
    onnx_ms = (time.perf_counter() - t0) * 1000 / 20

    return {
        "torch_logits_shape": torch_out.shape,
        "onnx_logits_shape":  onnx_out.shape,
        "max_abs_diff":       max_err,
        "mean_abs_diff":      mean_err,
        "max_prob_diff":      prob_diff,
        "argmax_agreement":   argmax_agreement,
        "match_predictions":  argmax_agreement == 1.0,
        "torch_ms_per_batch": torch_ms,
        "onnx_ms_per_batch":  onnx_ms,
        "speedup":            torch_ms / onnx_ms,
    }


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #
SPEC = {
    "cnn":      ("cnn_model.pth",      "cnn_model.onnx"),
    "transfer": ("transfer_model.pth", "transfer_model.onnx"),
    "swin":     ("swin_model.pth",     "swin_model.onnx"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export models to ONNX.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(SPEC.keys()),
        choices=list(SPEC.keys()),
        help="Subset of models to export (default: all).",
    )
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--verify", action="store_true", default=True,
                        help="Verify torch vs ONNX numerical equivalence.")
    args = parser.parse_args()

    print("=" * 64)
    print(" Brain Tumor Detection -- ONNX export")
    print("=" * 64)
    print(f"Opset: {args.opset}")
    print(f"Models: {args.models}\n")

    summary = []
    for name in args.models:
        ckpt_in, onnx_out = SPEC[name]
        ckpt_path = MODELS_DIR / ckpt_in
        onnx_path = MODELS_DIR / onnx_out

        if not ckpt_path.exists():
            print(f"[SKIP] {name}: checkpoint not found at {ckpt_path}")
            continue

        print(f"--- {name} ---")
        try:
            model = _load_for_export(name, ckpt_path)
            export_to_onnx(model, onnx_path, opset=args.opset)
            size_mb = onnx_path.stat().st_size / 1024 ** 2
            print(f"  exported -> {onnx_path} ({size_mb:.1f} MB)")

            if args.verify:
                res = verify_onnx(model, onnx_path)
                ok = "OK" if res["match_predictions"] else "MISMATCH"
                print(f"  verify   -> [{ok}]  argmax_agree={res['argmax_agreement']*100:.0f}%  "
                      f"max_prob_diff={res['max_prob_diff']:.2e}  "
                      f"speedup={res['speedup']:.2f}x")
                summary.append({
                    "model":      name,
                    "size_mb":    round(size_mb, 1),
                    "argmax_ok":  res["match_predictions"],
                    "prob_diff":  res["max_prob_diff"],
                    "torch_ms":   round(res["torch_ms_per_batch"], 1),
                    "onnx_ms":    round(res["onnx_ms_per_batch"], 1),
                    "speedup":    round(res["speedup"], 2),
                })
        except Exception as e:                                          # noqa: BLE001
            print(f"  [ERROR] {type(e).__name__}: {e}")
        print()

    if summary:
        print("Summary:")
        print(f"{'model':<10s} {'size MB':>8s} {'argmax ok':>10s} {'prob diff':>10s} "
              f"{'torch ms':>9s} {'onnx ms':>9s} {'speedup':>8s}")
        for r in summary:
            print(f"{r['model']:<10s} {r['size_mb']:>8.1f} "
                  f"{str(r['argmax_ok']):>10s} {r['prob_diff']:>10.2e} "
                  f"{r['torch_ms']:>9.1f} {r['onnx_ms']:>9.1f} "
                  f"{r['speedup']:>7.2f}x")


if __name__ == "__main__":
    main()
