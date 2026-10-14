"""Capture runtime and a real CUDA forward/backward benchmark, not a training result."""
import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    report = {'time_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
              'executable': sys.executable, 'platform': platform.platform(),
              'torch': torch.__version__, 'cuda_build': torch.version.cuda,
              'cuda_available': torch.cuda.is_available(),
              'disk': shutil.disk_usage(args.output.parent)._asdict()}
    report['nvidia_smi'] = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu', '--format=csv'], capture_output=True, text=True).stdout
    if torch.cuda.is_available():
        report['device'] = torch.cuda.get_device_name()
        report['capability'] = torch.cuda.get_device_capability()
        report['compiled_architectures'] = torch.cuda.get_arch_list()
        torch.manual_seed(42)
        model = torch.nn.Sequential(torch.nn.Linear(768, 256), torch.nn.GELU(), torch.nn.Linear(256, 5)).cuda()
        x = torch.randn(64, 32, 768, device='cuda')
        optimizer = torch.optim.AdamW(model.parameters())
        def step():
            optimizer.zero_grad(set_to_none=True)
            loss = model(x).square().mean()
            loss.backward()
            optimizer.step()
            return loss
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            loss = step()
        torch.cuda.synchronize()
        report['synthetic_environment_probe'] = {'steps': 50, 'seconds': time.perf_counter() - start,
            'final_loss_finite': bool(torch.isfinite(loss)), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'note': 'Synthetic CUDA environment check only; not a data/model performance result.'}
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
