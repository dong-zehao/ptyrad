from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ptyrad.optics.aberrations import Aberrations
from ptyrad.optics.constants import get_wavelength_ang
from ptyrad.optics.parametrized_probe import ParametrizedProbe, default_coefficients
from ptyrad.optics.probe import make_stem_probe
from ptyrad.optics.propagator import near_field_evolution
from ptyrad.params.model_params import ModelParams
from ptyrad.params.probe_params import ProbeParams
from ptyrad.core.models.parametrized import (
    ParametrizedPtychoModel, create_ptycho_model, prepare_parametrized_params,
)
from ptyrad.io.save import save_dict_to_hdf5
from ptyrad.io.load import load_ptyrad
from ptyrad.solver.reconstruction import toggle_grad_requires, create_optimizer


def generator(aberrations=None, **kwargs):
    args = dict(size=16, dx=.2, wavelength=get_wavelength_ang(80), conv_angle=25,
                intensity=100, aberrations=aberrations, dtype=torch.float64)
    args.update(kwargs)
    return ParametrizedProbe(**args)


@pytest.mark.parametrize("aberrations", [{}, {"C10": -100},
    {"C12a": 50, "C12b": -20, "C21a": 300},
    {"C30": 10000, "C56b": 100000, "C70": 200000}])
@pytest.mark.parametrize("z", [0, 25])
def test_numpy_equivalence(aberrations, z):
    probe = generator(aberrations, z_shift=z)()[0].detach().numpy()
    ref = make_stem_probe(80, 25, 16, .2, Aberrations(aberrations)) * 10
    if z:
        ref = np.fft.ifft2(np.fft.fft2(ref) * near_field_evolution((16, 16), .2, z, get_wavelength_ang(80)))
    np.testing.assert_allclose(probe, ref, atol=1e-10, rtol=1e-10)
    assert np.square(np.abs(probe)).sum() == pytest.approx(100)


def test_all_zero_coefficients_have_correct_gradients():
    probe = generator()
    assert len(probe.names) == 25
    values = torch.zeros(25, dtype=torch.float64, requires_grad=True)
    # Scale each direction so even very high-order terms give a measurable perturbation.
    scales = 1 / probe.basis.abs().amax(dim=(1, 2))
    fn = lambda x: torch.view_as_real(probe.from_values(x * scales))
    assert torch.autograd.gradcheck(fn, (values,), eps=1e-6, atol=1e-5, rtol=1e-4)
    for index in range(25):
        jac = torch.autograd.functional.jvp(fn, values, torch.eye(25, dtype=torch.float64)[index])[1]
        assert jac.abs().max() > 0


def fixture_model(overrides=None, start=1, end=None, state=None):
    torch.manual_seed(2)
    n = 16
    init_params = dict(probe_conv_angle=25, probe_aberrations={}, probe_z_shift=0)
    probe = generator(dtype=torch.float32)().detach().numpy()
    values = dict(probe=probe, obj=np.exp(1j * np.random.default_rng(2).normal(size=(1, 1, 20, 20))).astype('complex64'),
                  measurements=np.ones((2, n, n), dtype='float32'), obj_tilts=np.zeros((1, 2)),
                  slice_thickness=1., probe_pos_shifts=np.zeros((2, 2)), omode_occu=np.ones(1),
                  H=np.ones((n, n), dtype='complex64'), N_scan_slow=1, N_scan_fast=2,
                  crop_pos=np.array([[0, 0], [2, 2]]), dx=.2, dk=1/(n*.2),
                  lambd=get_wavelength_ang(80), random_seed=2, length_unit='Ang', scan_affine=None,
                  meas_Npix=n, simu_Npix=n, simu_match_mode='crop', recon_provenance={})
    params = ModelParams().model_dump()
    for update in params['update_params'].values():
        update.update(lr=0., start_iter=None)
    params['update_params']['probe'].update(lr=1., start_iter=start, end_iter=end)
    params['probe_params'] = dict(parametrize=True, coefficients=overrides or {})
    return ParametrizedPtychoModel(values, params, init_params, state=state), values, params, init_params


def test_defaults_overrides_and_schedule():
    model, _, _, _ = fixture_model({'C30': {'trainable': False}, 'C10': {'lr': 2.}, 'C12a': {'lr': 0.}}, start=2, end=4)
    assert len(model.optimizable_params) == 1
    assert model.probe_generator.coefficients.shape == (25,)
    assert len(list(model.probe_generator.parameters())) == 1
    assert model.lr_params['probe_coefficients'] == 2
    assert all('opt_probe' != name for name, _ in model.named_parameters())
    for iteration, active in [(1, False), (2, True), (3, True), (4, False)]:
        toggle_grad_requires(model, iteration)
        assert model.probe_generator.coefficients[model.probe_generator.names.index('C10')].requires_grad is active
        assert not model.probe_generator.trainable_mask[model.probe_generator.names.index('C30')]
        assert not model.probe_generator.trainable_mask[model.probe_generator.names.index('C12a')]


def test_reconstruction_backward_and_checkpoint(tmp_path):
    model, values, params, init_params = fixture_model({'C30': {'trainable': False}})
    indices = torch.tensor([0, 1])
    with torch.no_grad():
        model.probe_generator.coefficients[model.probe_generator.names.index('C10')].fill_(25)
        target = model(indices).detach()
        model.probe_generator.coefficients[model.probe_generator.names.index('C10')].zero_()
    optimizer = create_optimizer(model.optimizer_params, model.optimizable_params)
    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        loss = (model(indices) - target).square().mean()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0]
    assert model.probe_generator.coefficients[model.probe_generator.names.index('C30')].item() == 0
    assert model.get_complex_probe_view().shape == (1, 16, 16)
    assert model.get_complex_probe_view().abs().square().sum().item() == pytest.approx(100, rel=1e-5)

    path = tmp_path / 'probe.hdf5'
    save_dict_to_hdf5({'parametrized_probe': model.export_parametrized_probe(),
                       'optim_state_dict': optimizer.state_dict()}, str(path))
    state = load_ptyrad(str(path))['parametrized_probe']
    restored = ParametrizedPtychoModel(values, params, init_params, state=state)
    torch.testing.assert_close(restored.get_complex_probe_view(), model.get_complex_probe_view())
    restored_optimizer = create_optimizer({'name': 'Adam', 'configs': {}, 'load_state': str(path)},
                                          restored.optimizable_params)
    assert restored_optimizer.state  # Must not silently fall back to a fresh optimizer.
    for current, optim in [(model, optimizer), (restored, restored_optimizer)]:
        optim.zero_grad()
        (current(indices) - target).square().mean().backward()
        optim.step()
    torch.testing.assert_close(restored.get_complex_probe_view(), model.get_complex_probe_view())
    wrong = deepcopy(state)
    wrong['geometry']['dx'] *= 2
    with pytest.raises(ValueError, match='geometry mismatch'):
        ParametrizedPtychoModel(values, params, init_params, state=wrong)


def test_configuration_and_preparation():
    assert not ModelParams().probe_params.parametrize
    for name in ['C12', 'phi12', 'Cs', 'C10a', 'C11a']:
        with pytest.raises(ValueError):
            ProbeParams(coefficients={name: {}})
    for lr in [-1, float('nan'), float('inf')]:
        with pytest.raises(ValueError):
            ProbeParams(coefficients={'C10': {'lr': lr}})
    params = dict(model_params={'probe_params': {'parametrize': True}},
                  init_params={'probe_pmode_max': 4},
                  constraint_params={key: {'start_iter': 1} for key in
                      ['probe_mask_k', 'probe_mask_r', 'obj_z_recenter', 'ortho_pmode', 'fix_probe_int']})
    prepare_parametrized_params(params)
    assert params['init_params']['probe_pmode_max'] == 1
    assert all(c['start_iter'] is None for c in params['constraint_params'].values())
    params['init_params']['probe_source'] = 'custom'
    with pytest.raises(ValueError):
        prepare_parametrized_params(params)


@pytest.mark.parametrize('name,configs', [('AdamW', {'weight_decay': .3}), ('SGD', {'momentum': .9, 'weight_decay': .3})])
def test_vector_updates_match_individual_rates(name, configs):
    model, _, _, _ = fixture_model({'C30': {'trainable': False}, 'C10': {'lr': 2.}})
    vector = model.probe_generator.coefficients
    with torch.no_grad():
        vector.copy_(torch.linspace(1, 2, vector.numel()))
    reference = [torch.nn.Parameter(value.clone().reshape(1)) for value in vector.detach()]
    rates = [0 if n == 'C30' else 2 if n == 'C10' else 1 for n in model.probe_generator.names]
    actual = create_optimizer({'name': name, 'configs': configs}, model.optimizable_params)
    expected = getattr(torch.optim, name)([{'params': [p], 'lr': rate} for p, rate in zip(reference, rates) if rate], **configs)
    schedulers = [torch.optim.lr_scheduler.StepLR(o, step_size=1, gamma=.5) for o in (actual, expected)]
    for _ in range(3):
        actual.zero_grad()
        expected.zero_grad()
        vector.square().sum().backward()
        sum(p.square().sum() for p, rate in zip(reference, rates) if rate).backward()
        actual.step()
        expected.step()
        for scheduler in schedulers:
            scheduler.step()
        torch.testing.assert_close(vector, torch.cat(reference), rtol=1e-5, atol=1e-6)
    assert len(actual.param_groups) == 1
    assert len(actual.state) == 1


def test_compiled_generator():
    probe = generator(dtype=torch.float32)
    compiled = torch.compile(probe, backend='aot_eager', fullgraph=True)
    torch.testing.assert_close(compiled(), probe())
    compiled().real.square().sum().backward()
    assert all(p.grad is not None for p in probe.parameters())


def test_compiled_optimizer_preserves_vector_overrides():
    model, _, _, _ = fixture_model({'C30': {'trainable': False}, 'C10': {'lr': 2.}})
    optim = create_optimizer({'name': 'AdamW', 'configs': {'weight_decay': .1}}, model.optimizable_params)
    optim.step = torch.compile(optim.step, backend='aot_eager')
    vector = model.probe_generator.coefficients
    frozen = model.probe_generator.names.index('C30')
    with torch.no_grad():
        vector.fill_(1.)
    for _ in range(2):
        optim.zero_grad()
        vector.square().sum().backward()
        optim.step()
    assert vector[frozen].item() == 1.
    assert vector[0].item() != 1.


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_forward_backward():
    cpu = generator(dtype=torch.float32)
    gpu = generator(dtype=torch.float32, device='cuda')
    torch.testing.assert_close(gpu().cpu(), cpu(), atol=1e-5, rtol=1e-5)
    weight = torch.randn_like(cpu().real)
    (cpu().real * weight).sum().backward()
    (gpu().real * weight.cuda()).sum().backward()
    torch.testing.assert_close(gpu.coefficients.grad.cpu(), cpu.coefficients.grad,
                               atol=1e-6, rtol=1e-4)


def test_high_order_fixed_unless_selected():
    model, values, params, init = fixture_model()
    init['probe_aberrations'] = {'C70': 1000}
    fixed = ParametrizedPtychoModel(values, params, init)
    assert not fixed.probe_generator.trainable_mask[fixed.probe_generator.names.index('C70')]
    params['probe_params']['coefficients']['C70'] = {'lr': 5}
    active = ParametrizedPtychoModel(values, params, init)
    assert active.probe_generator.trainable_mask[active.probe_generator.names.index('C70')]


def test_disabled_factory_keeps_pixel_model():
    _, values, params, init = fixture_model()
    params['probe_params']['parametrize'] = False
    result = create_ptycho_model(SimpleNamespace(init_variables=values, init_params=init),
                                {'model_params': params})
    assert type(result).__name__ == 'PtychoModel'
    assert isinstance(result.opt_probe, torch.nn.Parameter)


def test_solver_initialization_and_saved_results(minimal_params_dict, tmp_path):
    import yaml
    from ptyrad.params import load_params
    from ptyrad.solver.ptyrad_solver import PtyRADSolver

    raw = deepcopy(minimal_params_dict)
    np.ones((4, 8, 8), dtype='float32').tofile(raw['init_params']['meas_params']['path'])
    raw['init_params']['probe_aberrations'] = {'C10': .000123456789}
    raw['init_params']['probe_interpolate'] = None
    raw['model_params'] = {'probe_params': {'parametrize': True}}
    raw['recon_params'].update(NITER=2, BATCH_SIZE={'size': 2}, selected_figs=[], save_result=['model', 'optim_state'])
    config = tmp_path / 'params.yaml'
    config.write_text(yaml.safe_dump(raw), encoding='utf8')
    params = load_params(str(config))
    assert params['init_params']['probe_aberrations']['C10'] == .000123456789
    solver = PtyRADSolver(params, device='cpu')
    assert solver.init.init_variables['probe'].shape[0] == 1
    assert params['init_params']['probe_pmode_max'] == 4  # Caller config is untouched.
    solver.reconstruct()
    paths = sorted(tmp_path.rglob('model_iter*.hdf5'))
    assert len(paths) == 2
    saved = load_ptyrad(str(paths[-1]))
    assert len(saved['parametrized_probe']['coefficients']) == 25
    assert saved['optimizable_tensors']['probe'].shape == (1, 8, 8)
    resumed_params = deepcopy(solver.params)
    resumed_params['init_params'].update(probe_source='PtyRAD', probe_params=str(paths[-1]))
    resumed_params['model_params']['optimizer_params']['load_state'] = str(paths[-1])
    resumed = PtyRADSolver(resumed_params, device='cpu')
    resumed_model = create_ptycho_model(resumed.init, resumed.params)
    torch.testing.assert_close(resumed_model.get_complex_probe_view(), solver.reconstruct_results.get_complex_probe_view())
