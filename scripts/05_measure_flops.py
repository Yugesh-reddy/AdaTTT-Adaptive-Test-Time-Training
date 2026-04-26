"""
Measure actual FLOPs of trainable components using torch.profiler.

Compares measured values against hardcoded constants in AdaptiveRouter
to validate FLOP accounting accuracy.

Usage:
    python scripts/05_measure_flops.py [--config config/config.yaml]
"""

import argparse
import sys
import os

import torch
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttt.models import FusionModule, ConfidenceGate, PredictionHead
from ttt.gate import AdaptiveRouter
from ttt.utils import load_config


def measure_flops(module, args, label):
    """Profile a module's forward pass and return total FLOPs."""
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            module(*args)

    with profile(
        activities=[ProfilerActivity.CPU],
        with_flops=True,
        record_shapes=True,
    ) as prof:
        with torch.no_grad():
            module(*args)

    total_flops = sum(
        evt.flops for evt in prof.key_averages() if evt.flops is not None and evt.flops > 0
    )
    return total_flops


def main():
    parser = argparse.ArgumentParser(description="Measure FLOPs of trainable components")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    dim = config.get("fusion_dim", 768)
    num_heads = config.get("fusion_heads", 12)
    num_layers = config.get("fusion_layers", 2)
    dropout = config.get("fusion_dropout", 0.1)
    pred_hidden = config.get("prediction_hidden", 1024)
    num_answers = config.get("num_answers", 3129)
    gate_hidden = config.get("gate_hidden", 256)

    # Build modules on CPU
    fusion = FusionModule(dim, num_heads, num_layers, dropout).eval()
    gate = ConfidenceGate(dim, gate_hidden).eval()
    pred_head = PredictionHead(dim, pred_hidden, num_answers).eval()

    # Dummy inputs (batch=1)
    visual = torch.randn(1, 197, dim)
    text = torch.randn(1, 20, dim)
    text_mask = torch.ones(1, 20, dtype=torch.bool)
    z = torch.randn(1, dim)

    # Measure
    fusion_flops = measure_flops(fusion, (visual, text, text_mask), "Fusion")
    gate_flops = measure_flops(gate, (z,), "Gate")
    pred_flops = measure_flops(pred_head, (z,), "PredictionHead")

    # Hardcoded constants from AdaptiveRouter
    hardcoded_fusion = AdaptiveRouter.FUSION_FLOPS
    hardcoded_pred = AdaptiveRouter.PRED_FLOPS

    print("=" * 65)
    print("FLOP Measurement Report")
    print("=" * 65)
    print(f"{'Component':<25} {'Measured':>15} {'Hardcoded':>15}")
    print("-" * 65)
    print(f"{'Fusion (fwd)':<25} {fusion_flops/1e9:>12.3f} G  {hardcoded_fusion/1e9:>12.3f} G")
    print(f"{'PredictionHead (fwd)':<25} {pred_flops/1e9:>12.3f} G  {hardcoded_pred/1e9:>12.3f} G")
    print(f"{'ConfidenceGate (fwd)':<25} {gate_flops/1e9:>12.3f} G  {'(not tracked)':>15}")
    print("-" * 65)

    # TTT step estimate: fusion fwd + bwd (~2x) + pred fwd + bwd (~2x)
    ttt_step_est = 3 * fusion_flops + 3 * pred_flops  # fwd+bwd ≈ 3x fwd
    hardcoded_ttt = AdaptiveRouter.TTT_STEP_FLOPS
    print(f"{'TTT step (est 3x fwd)':<25} {ttt_step_est/1e9:>12.3f} G  {hardcoded_ttt/1e9:>12.3f} G")

    # Consistency overhead
    hardcoded_cons_oh = AdaptiveRouter.CONSISTENCY_OVERHEAD_FLOPS
    hardcoded_cons_ps = AdaptiveRouter.CONSISTENCY_PER_STEP_FLOPS
    print(f"{'Consistency overhead':<25} {'(2x ViT)':>15}  {hardcoded_cons_oh/1e9:>12.3f} G")
    print(f"{'Consistency per-step':<25} {'(2x fusion)':>15}  {hardcoded_cons_ps/1e9:>12.3f} G")
    print("=" * 65)

    # Deviation check
    deviations = []
    if hardcoded_fusion > 0:
        dev = abs(fusion_flops - hardcoded_fusion) / hardcoded_fusion
        deviations.append(("Fusion", dev, fusion_flops, hardcoded_fusion))
    if hardcoded_pred > 0:
        dev = abs(pred_flops - hardcoded_pred) / hardcoded_pred
        deviations.append(("PredictionHead", dev, pred_flops, hardcoded_pred))

    print("\nDeviation Analysis:")
    any_large = False
    for name, dev, measured, hardcoded in deviations:
        if dev < 0.25:
            status = "OK"
        elif dev < 1.0:
            status = "REVIEW"
        else:
            status = "MISMATCH"
            any_large = True
        print(f"  {name}: measured {measured/1e9:.3f}G vs hardcoded {hardcoded/1e9:.3f}G "
              f"({dev*100:.1f}% deviation) [{status}]")

    if any_large:
        print("\nNote: Large deviations detected. The hardcoded FLOP constants")
        print("in AdaptiveRouter may need updating. The profiler measures actual")
        print("multiply-accumulate ops; hardcoded values are rough estimates.")
        print("Consider updating gate.py constants to match measured values.")

    # Parameter counts
    print("\nParameter Counts:")
    print(f"  Fusion:         {sum(p.numel() for p in fusion.parameters()):>12,}")
    print(f"  Gate:           {sum(p.numel() for p in gate.parameters()):>12,}")
    print(f"  PredictionHead: {sum(p.numel() for p in pred_head.parameters()):>12,}")
    total = sum(p.numel() for m in [fusion, gate, pred_head] for p in m.parameters())
    print(f"  Total trainable:{total:>12,}")


if __name__ == "__main__":
    main()
# Update FLOP measurement for cached features
