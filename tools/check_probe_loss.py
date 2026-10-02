"""Check the installed loss implementation against a saved probe (no data needed).

Run in the reconstruction environment, optionally with --device cuda --compile.
The checkpoint is opened read-only. No reconstruction or optimizer state is changed.
"""

import argparse
import inspect
import json
from types import SimpleNamespace

import h5py
import torch

from ptyrad.core.losses import CombinedLoss
from ptyrad.params.loss_params import LossParams


def read_group(group):
    result = {}
    for name, value in group.items():
        if isinstance(value, h5py.Group):
            result[name] = read_group(value)
        else:
            value = value[()]
            if isinstance(value, bytes):
                value = value.decode()
            elif hasattr(value, 'tolist'):
                value = value.tolist()
            result[name] = None if isinstance(value, str) and value == '__NONE__' else value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--backend', default='inductor')
    args = parser.parse_args()
    print(f'Loaded CombinedLoss from {inspect.getfile(CombinedLoss)}', flush=True)
    with h5py.File(args.checkpoint, 'r') as handle:
        probe = torch.tensor(handle['optimizable_tensors/probe'][()], device=args.device)
        dx = float(handle['model_attributes/dx'][()])
        wavelength = float(handle['model_attributes/lambd'][()])
        init = read_group(handle['params/init_params'])
        constraints = read_group(handle['params/constraint_params'])
        params = LossParams(**read_group(handle['params/loss_params'])).model_dump()
        saved = float(handle['avg_losses/loss_probe_reg'][()])
    if not params['loss_probe_reg']['state'] or params['loss_probe_reg']['weight'] == 0:
        raise RuntimeError('Checkpoint does not enable a nonzero probe regularization weight')
    # Match the real model's real-valued storage and complex view.
    storage = torch.view_as_real(probe).clone().requires_grad_()
    probe = torch.view_as_complex(storage)
    model = SimpleNamespace(get_complex_probe_view=lambda: probe, dx=dx, lambd=wavelength)
    loss_fn = CombinedLoss(params, device=args.device)
    loss_fn.configure_probe_reg(model, init, constraints)
    dp = torch.ones(1, 4, 4, device=args.device)
    patches = torch.zeros(1, 1, 1, 2, 2, device=args.device)
    occupancy = torch.ones(1, device=args.device)
    index = list(params).index('loss_probe_reg')

    def evaluate(value):
        total, terms = loss_fn(dp * 2, dp, patches, patches, occupancy, torch.view_as_complex(value))
        if len(terms) != len(params):
            raise RuntimeError('Configured probe loss is missing from forward(); update installed PtyRAD')
        return total, terms[index]

    total, regularizer = evaluate(storage)
    gradient = torch.autograd.grad(total, storage)[0]
    torch.testing.assert_close(regularizer, loss_fn.get_loss_probe_reg(probe))
    if not torch.isfinite(gradient).all() or gradient.norm() == 0:
        raise RuntimeError('Probe loss has a zero or non-finite gradient on this checkpoint')
    result = dict(torch_version=torch.__version__, device=args.device,
                  saved_batch_average=saved, checkpoint_weighted_loss=regularizer.item(),
                  probe_gradient_norm=gradient.norm().item())
    if args.compile:
        compiled_total, compiled_reg = torch.compile(evaluate, backend=args.backend)(storage)
        compiled_grad = torch.autograd.grad(compiled_total, storage)[0]
        torch.testing.assert_close(compiled_reg, regularizer)
        torch.testing.assert_close(compiled_grad, gradient, rtol=1e-4, atol=1e-7)
        result.update(compiled_weighted_loss=compiled_reg.item(), compiled_matches_eager=True)
    print(json.dumps(result, indent=2))
    print('PASS: forward includes the probe regularizer and its gradient.')


if __name__ == '__main__':
    main()
