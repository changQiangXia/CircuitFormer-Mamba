from pathlib import Path
import json
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from true_mamba_experiments.modules.true_mamba_neck import TrueMambaNeckPrototype


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the current true Mamba module smoke test.")

    device = torch.device("cuda")
    torch.manual_seed(3407)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    module = TrueMambaNeckPrototype(
        dim=64,
        d_state=16,
        d_conv=4,
        expand=2,
        downsample=1,
        bidirectional=True,
        out_proj_init_zero=True,
    ).to(device)
    module.train()

    x = torch.randn(2, 64, 256, 256, device=device, requires_grad=True)

    torch.cuda.synchronize(device)
    start = time.perf_counter()
    y = module(x)
    loss = y.mean()
    torch.cuda.synchronize(device)
    forward_time = time.perf_counter() - start

    start = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(device)
    backward_time = time.perf_counter() - start

    info = {
        "module": module.__class__.__name__,
        "input_shape": list(x.shape),
        "output_shape": list(y.shape),
        "dtype": str(y.dtype),
        "device": str(y.device),
        "param_count": sum(p.numel() for p in module.parameters()),
        "forward_time_sec": round(forward_time, 6),
        "backward_time_sec": round(backward_time, 6),
        "peak_memory_mb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2),
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
