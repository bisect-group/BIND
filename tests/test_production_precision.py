"""Production precision (fp16 autocast with GradScaler, the original train_graph.py setting) against fp32.

Both paths train the same model on the same toy rows in the same order for 3 epochs of 2 optimizer steps
(dropout 0, AdamW lr 1e-4), then evaluate. The cache stores fp16 states in both paths, so only the compute
precision differs. The bound is the noise-control ratio (K times the fp16 run perturbed by one weight rounding),
as in the test docstring. EQUIV_NEGATIVE_CONTROL=1 scales the production weights by 1.05; the test must then fail.
"""

import os

import numpy as np
import pytest
import torch

from emulator_bench.modeling import build_model
from emulator_bench.train_regression_tvt import evaluate_loader, train_one_epoch
from test_cache_and_loader import WIDTH, make_loader, toy_cache  # noqa: F401  (toy_cache is a fixture)

EPOCHS = 3
K = 10  # the production gap may be 10 times the fp32 noise (the fp16 run with initial weights perturbed by one fp16 rounding) before it counts as a defect


U = 2**-11  # one rounding of the weights at the production precision (fp16)


def perturb(module, rel=U):
    generator = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(rel * p.abs() * torch.randn(p.shape, generator=generator).to(p.device))


def train(cache, device, autocast_dtype, noise=False, scale=1.0):
    """Loss of every optimizer step (each step is one call on one batch) and the final RMSE."""
    torch.manual_seed(0)
    model = build_model(np.random.default_rng(0).random((33, WIDTH)).astype(np.float32), cross_attention_dropout=0.0).to(device)
    if noise:
        perturb(model)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    # The default scale of 2**16 overflows the first four steps of this model (the scaler skips them and halves
    # the scale); a 6-step toy run would then compare a trained model with an untrained one.
    scaler = torch.amp.GradScaler("cuda", init_scale=2**10, enabled=autocast_dtype == torch.float16)
    criterion = torch.nn.HuberLoss(delta=2.0)
    loader, _ = make_loader(cache, micro_batch_size=1, batch_size=2)
    batches = list(loader)
    losses = [
        train_one_epoch(model, [batch], optimizer, criterion, device, scaler, "ic50", 1, 10**6, autocast_dtype=autocast_dtype, log_every_steps=10**6)["loss"]
        for _ in range(EPOCHS)
        for batch in batches
    ]
    *_, metrics, _ = evaluate_loader(model, batches, device, criterion, "ic50", autocast_dtype, show_progress=False)
    assert not scaler.is_enabled() or scaler.get_scale() == 2**10, "the scaler skipped a step"
    return torch.tensor(losses, dtype=torch.float64), metrics["rmse"]


@pytest.mark.gpu
def test_fp16_autocast_training_matches_fp32(toy_cache):  # noqa: F811
    """Production fp16 autocast with GradScaler against fp32 on the same rows, seed and order.

    The reference is fp32; the noise control is the fp16 run with initial weights perturbed by 2**-11 relative.
    Bounds: max per-step loss gap and final RMSE gap each within K times their noise floor; the step-0
    loss gap within K times its floor like every other step. The scaler starts at 2**10 and no step is skipped.
    """
    device = torch.device("cuda")
    (ref, ref_rmse), (noise, noise_rmse), (prod, prod_rmse) = (
        train(toy_cache, device, None),
        train(toy_cache, device, torch.float16, noise=True),
        train(toy_cache, device, torch.float16),
    )
    cand, cand_rmse = (prod, prod_rmse)
    if os.environ.get("EQUIV_NEGATIVE_CONTROL") == "1":
        cand, cand_rmse = train(toy_cache, device, torch.float16, scale=1.05)
    floor, gap = (noise - prod).abs().max().item(), (cand - ref).abs().max().item()
    tiny, tiny0 = 1e-7 * ref.abs().max().item(), 1e-7 * ref[0].abs().item()
    gap0, floor0 = (cand[0] - ref[0]).abs().item(), (noise[0] - prod[0]).abs().item()
    final_gap, final_floor = abs(cand_rmse - ref_rmse), abs(noise_rmse - prod_rmse)
    print(
        f"bind fp16: max gap={gap:.3e}, max floor={floor:.3e}, ratio={gap / max(floor, tiny):.2f}, K={K}, "
        f"step0 gap={gap0:.3e}, floor={floor0:.3e}, ratio={gap0 / max(floor0, tiny0):.2f}; final gap={final_gap:.3e}, final floor={final_floor:.3e}; ref={ref.tolist()} prod={cand.tolist()}"
    )
    assert gap0 <= K * max(floor0, tiny0)
    assert gap <= K * max(floor, tiny)
    assert final_gap <= K * max(final_floor, 1e-7 * abs(ref_rmse))
