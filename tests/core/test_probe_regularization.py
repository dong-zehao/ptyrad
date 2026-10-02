from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from ptyrad.core.losses import CombinedLoss
from ptyrad.core.probe_regularization import (
    interior_aperture_mask,
    mixed_curvature,
    product_curvature,
)
from ptyrad.params.loss_params import LossParams
from ptyrad.solver.reconstruction import compute_loss, recon_step


def _probe(nmodes=3, shape=(32, 32)):
    generator = torch.Generator().manual_seed(13)
    real = torch.randn((nmodes, *shape), generator=generator, dtype=torch.float64)
    imag = torch.randn((nmodes, *shape), generator=generator, dtype=torch.float64)
    return torch.complex(real, imag)


def _mask(shape=(32, 32), dx=0.5, conv_angle=12.0):
    return interior_aperture_mask(shape, dx, 0.02, conv_angle, 0.85, 'cpu')


def _manual_curvature(probe, mask, mixed):
    pupil = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(probe, dim=(-2, -1)), norm='ortho'),
        dim=(-2, -1),
    )
    numerator = 0.0
    denominator = 0.0
    for y in range(1, mask.shape[0] - 1):
        for x in range(1, mask.shape[1] - 1):
            for dy, dx in ((1, 0), (0, 1)):
                a, b, c = (y - dy, x - dx), (y, x), (y + dy, x + dx)
                if not (mask[a] and mask[b] and mask[c]):
                    continue
                if mixed:
                    qi, qj, qk = pupil[:, a[0], a[1]], pupil[:, b[0], b[1]], pupil[:, c[0], c[1]]
                    si, sj, sk = qi.abs().square().sum(), qj.abs().square().sum(), qk.abs().square().sum()
                    weight = sj * (si * sk).sqrt()
                    cij = (qi * qj.conj()).sum()
                    cjk = (qj * qk.conj()).sum()
                    numerator += (weight - (cij * cjk.conj()).real).item()
                    denominator += weight.item()
                else:
                    qi, qj, qk = pupil[0, a[0], a[1]], pupil[0, b[0], b[1]], pupil[0, c[0], c[1]]
                    endpoints, midpoint = qi * qk, qj.square()
                    numerator += (endpoints - midpoint).abs().square().item()
                    denominator += (endpoints.abs().square() + midpoint.abs().square()).item()
    return numerator / denominator


@pytest.mark.parametrize('mixed', [False, True])
def test_curvature_matches_explicit_formula_and_has_finite_gradient(mixed):
    probe = _probe().requires_grad_()
    mask = _mask()
    metric = mixed_curvature(probe, mask) if mixed else product_curvature(probe, mask)
    assert metric.item() == pytest.approx(_manual_curvature(probe.detach(), mask, mixed), rel=1e-12)
    metric.backward()
    assert torch.isfinite(probe.grad).all()
    assert probe.grad.abs().sum() > 0


@pytest.mark.parametrize('metric', [product_curvature, mixed_curvature])
def test_curvature_ignores_global_complex_scale_and_integer_translation(metric):
    probe = _probe()
    mask = _mask()
    baseline = metric(probe, mask)
    assert torch.allclose(metric(probe * (2.4 + 1.2j), mask), baseline, atol=1e-12)
    shifted = torch.roll(probe, shifts=(2, -3), dims=(-2, -1))
    assert torch.allclose(metric(shifted, mask), baseline, atol=1e-12)


def test_mixed_curvature_is_invariant_to_unitary_mode_rotation():
    probe = _probe()
    mask = _mask()
    matrix = _probe(nmodes=1, shape=(3, 3))[0]
    unitary, _ = torch.linalg.qr(matrix)
    rotated = (unitary @ probe.reshape(3, -1)).reshape_as(probe)
    assert torch.allclose(mixed_curvature(rotated, mask), mixed_curvature(probe, mask), atol=1e-12)


def test_mask_rejects_no_valid_triples():
    with pytest.raises(ValueError, match='three-pixel triples'):
        _mask(conv_angle=0.1)


def _model(probe, dx=0.5):
    return SimpleNamespace(
        get_complex_probe_view=lambda: probe,
        dx=torch.tensor(dx),
        lambd=torch.tensor(0.02),
    )


def _loss_params(mode='primary', state=True):
    params = LossParams().model_dump()
    params['loss_probe_reg'].update(state=state, mode=mode)
    return params


@pytest.mark.parametrize('mode', ['primary', 'mixed'])
def test_configure_and_compute_loss_includes_recorded_probe_term(mode):
    probe = _probe().requires_grad_()
    params = _loss_params(mode)
    loss_fn = CombinedLoss(params, device='cpu')
    init = {'probe_illum_type': 'electron', 'probe_conv_angle': 12.0}
    constraints = {'ortho_pmode': {'start_iter': 1, 'step': 1, 'end_iter': None}}
    loss_fn.configure_probe_reg(_model(probe), init, constraints)

    class ForwardModel:
        omode_occu = torch.tensor([1.0])

        def __call__(self, batch):
            self._current_object_patches = (torch.ones(1, 1, 1, 2, 2), torch.zeros(1, 1, 1, 2, 2))
            return torch.full((1, 4, 4), 2.0)

        def get_complex_probe_view(self):
            return probe

    model = ForwardModel()
    total, losses = compute_loss(None, model, model, torch.ones(1, 4, 4), loss_fn)
    assert len(losses) == len(params)
    assert total == sum(losses)
    assert losses[-1] > 0
    total.backward()
    assert torch.isfinite(probe.grad).all()


def test_disabled_loss_keeps_previous_total_and_needs_no_probe():
    params = _loss_params(state=False)
    loss_fn = CombinedLoss(params, device='cpu')
    dp = torch.ones(1, 4, 4)
    patches = torch.zeros(1, 1, 1, 2, 2)
    total, losses = loss_fn(dp * 2, dp, patches, patches, torch.ones(1))
    assert len(losses) == len(params)
    assert losses[-1] == 0
    assert total == sum(losses[:-1])

    old_params = dict(params)
    del old_params['loss_probe_reg']
    old_total, old_losses = CombinedLoss(old_params, device='cpu')(
        dp * 2, dp, patches, patches, torch.ones(1)
    )
    assert len(old_losses) == len(old_params)
    assert old_total == total


def test_multimode_primary_requires_every_iteration_orthogonalization():
    loss_fn = CombinedLoss(_loss_params(), device='cpu')
    with pytest.raises(ValueError, match='ortho_pmode'):
        loss_fn.configure_probe_reg(_model(_probe()),
                                    {'probe_conv_angle': 12.0},
                                    {'ortho_pmode': {'start_iter': None, 'step': 1, 'end_iter': None}})


def test_enabled_loss_requires_electron_calibration_and_refreshes_mask():
    loss_fn = CombinedLoss(_loss_params(mode='mixed'), device='cpu')
    model = _model(_probe())
    with pytest.raises(ValueError, match='electron'):
        loss_fn.configure_probe_reg(model, {'probe_illum_type': 'xray'}, {})
    loss_fn.configure_probe_reg(model, {'probe_conv_angle': 12.0}, {})
    original = loss_fn._probe_reg_mask.clone()
    loss_fn.configure_probe_reg(model, {'probe_conv_angle': 6.0}, {})
    assert not torch.equal(loss_fn._probe_reg_mask, original)
    changed_angle = loss_fn._probe_reg_mask.clone()
    loss_fn.configure_probe_reg(_model(_probe(), dx=1.0), {'probe_conv_angle': 6.0}, {})
    assert not torch.equal(loss_fn._probe_reg_mask, changed_angle)


def test_schema_rejects_invalid_probe_settings():
    for invalid in ({'mode': 'other'}, {'weight': -1}, {'aperture_fraction': 0}, {'aperture_fraction': 1.1}):
        with pytest.raises(ValueError):
            LossParams(loss_probe_reg=invalid)


def test_compiled_metric_matches_eager_and_backpropagates():
    probe = _probe().requires_grad_()
    mask = _mask()
    compiled = torch.compile(lambda value: product_curvature(value, mask), backend='eager')
    result = compiled(probe)
    assert torch.allclose(result, product_curvature(probe, mask))
    result.backward()
    assert torch.isfinite(probe.grad).all()


def test_reconstruction_step_records_probe_loss():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.opt_probe = torch.nn.Parameter(torch.view_as_real(_probe(nmodes=1).to(torch.complex64)).clone())
            self.device = 'cpu'
            self.lr_params = {'probe': 1e-3}
            self.lr_iters = defaultdict(list)
            self.avg_tilt_iters = defaultdict(list)
            self.loss_iters, self.iter_times, self.dz_iters = [], [], []
            self.opt_slice_thickness = torch.tensor(1.0)
            self.opt_obj_tilts = torch.zeros(1, 2)
            self.omode_occu = torch.ones(1)

        def get_complex_probe_view(self):
            return torch.view_as_complex(self.opt_probe)

        def get_measurements(self, indices):
            return torch.ones(len(indices), 4, 4)

        def forward(self, indices):
            self._current_object_patches = (torch.ones(1, 1, 1, 2, 2), torch.zeros(1, 1, 1, 2, 2))
            return torch.full((len(indices), 4, 4), 2.0)

        def clear_cache(self):
            self._current_object_patches = None

    model = TinyModel()
    loss_fn = CombinedLoss(_loss_params(), device='cpu')
    loss_fn.configure_probe_reg(_model(model.get_complex_probe_view()), {'probe_conv_angle': 12.0}, {})
    optimizer = torch.optim.SGD([model.opt_probe], lr=1e-3)
    recorded = recon_step([torch.tensor([0])], 1, model, optimizer, None, loss_fn,
                          lambda _model, _iteration: None, 1, 1, compute_loss_fn=compute_loss)
    assert recorded['loss_probe_reg'][0] > 0
    assert len(model.loss_iters) == 1
