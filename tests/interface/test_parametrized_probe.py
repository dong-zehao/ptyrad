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
from ptyrad.plotting.model import plot_probe_coefficient_curves, plot_summary
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
    params['probe_params'] = dict(parametrize=True,
                                  coefficients=overrides or {})
    return ParametrizedPtychoModel(values, params, init_params, state=state), values, params, init_params


def uniform_rate_overrides():
    return {name: {'lr': 1.} for name in default_coefficients()}


def test_normalized_coordinates_balance_phase_and_gradient_scales():
    probe = generator(dtype=torch.float64)
    for name in ('C10', 'C30', 'C50'):
        index = probe.names.index(name)
        normalized_basis = probe.basis[index] / probe.phase_per_angstrom[index]
        n = int(name[1])
        # At a common pupil radius, relative strengths depend on radius, not alpha**n.
        k = torch.fft.fftshift(torch.fft.fftfreq(16, d=.2, dtype=torch.float64))
        radius = k.abs() * probe.geometry['wavelength'] / .025
        torch.testing.assert_close(normalized_basis[8], radius ** (n + 1))
    q = probe.normalized_coefficients.detach().clone().requires_grad_()
    assert torch.autograd.gradcheck(
        lambda x: torch.view_as_real(probe.from_values(x / probe.phase_per_angstrom)),
        (q,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_default_and_override_rates_are_in_phase_radians():
    model, _, _, _ = fixture_model({'C30': {'lr': 7.}, 'C50': {'lr': 0.}})
    group = model.optimizable_params[0]
    scales = dict(zip(model.probe_generator.names, group['probe_update_scale']))
    assert group['lr'] == 7.
    assert scales['C10'] == pytest.approx(1 / 7)
    assert scales['C30'] == pytest.approx(1.)
    assert scales['C32a'] == pytest.approx(1 / 7)
    assert scales['C50'] == 0


def test_large_physical_coefficient_updates_without_roundoff_stall():
    model, values, params, init = fixture_model()
    init['probe_aberrations'] = {'C50': -1.85e7}
    params['update_params']['probe']['lr'] = 1e-4
    model = ParametrizedPtychoModel(values, params, init)
    gen = model.probe_generator
    i = gen.names.index('C50')
    before = gen.current_coefficients().detach().clone()
    opt = create_optimizer(model.optimizer_params, model.optimizable_params)
    gen.normalized_coefficients.grad = torch.ones_like(gen.normalized_coefficients)
    opt.step()
    change = gen.current_coefficients().detach() - before
    assert abs(change[i].item()) > 1000
    assert change[i].item() == pytest.approx(-1e-4 / gen.phase_per_angstrom[i].item(), rel=.002)



def test_defaults_overrides_and_schedule():
    model, _, _, _ = fixture_model({'C30': {'trainable': False}, 'C10': {'lr': 2.}, 'C12a': {'lr': 0.}}, start=2, end=4)
    assert len(model.optimizable_params) == 1
    assert model.probe_generator.normalized_coefficients.shape == (25,)
    assert len(list(model.probe_generator.parameters())) == 1
    assert model.lr_params['probe_normalized_coefficients'] == 2
    assert all('opt_probe' != name for name, _ in model.named_parameters())
    for iteration, active in [(1, False), (2, True), (3, True), (4, False)]:
        toggle_grad_requires(model, iteration)
        assert model.probe_generator.normalized_coefficients[model.probe_generator.names.index('C10')].requires_grad is active
        assert not model.probe_generator.trainable_mask[model.probe_generator.names.index('C30')]
        assert not model.probe_generator.trainable_mask[model.probe_generator.names.index('C12a')]


def test_reconstruction_backward_and_checkpoint(tmp_path):
    overrides = {name: {'lr': .01} for name in default_coefficients()}
    overrides['C30'] = {'trainable': False}
    model, values, params, init_params = fixture_model(overrides)
    indices = torch.tensor([0, 1])
    with torch.no_grad():
        model.probe_generator.normalized_coefficients[model.probe_generator.names.index('C10')].fill_(.25)
        target = model(indices).detach()
        model.probe_generator.normalized_coefficients[model.probe_generator.names.index('C10')].zero_()
    optimizer = create_optimizer(model.optimizer_params, model.optimizable_params)
    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        loss = (model(indices) - target).square().mean()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0]
    assert model.probe_generator.normalized_coefficients[model.probe_generator.names.index('C30')].item() == 0
    assert model.get_complex_probe_view().shape == (1, 16, 16)
    assert model.get_complex_probe_view().abs().square().sum().item() == pytest.approx(100, rel=1e-5)

    path = tmp_path / 'probe.hdf5'
    old_optimizer_state = deepcopy(optimizer.state_dict())
    old_optimizer_state['param_groups'][0]['probe_update_scale'] = [0.] * len(model.probe_generator.names)
    save_dict_to_hdf5({'parametrized_probe': model.export_parametrized_probe(),
                       'optim_state_dict': old_optimizer_state}, str(path))
    state = load_ptyrad(str(path))['parametrized_probe']
    restored = ParametrizedPtychoModel(values, params, init_params, state=state)
    torch.testing.assert_close(restored.get_complex_probe_view(), model.get_complex_probe_view())
    restored_optimizer = create_optimizer({'name': 'Adam', 'configs': {}, 'load_state': str(path)},
                                          restored.optimizable_params)
    assert restored_optimizer.state  # Must not silently fall back to a fresh optimizer.
    assert restored_optimizer.param_groups[0]['probe_update_scale'] == model.optimizable_params[0]['probe_update_scale']
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
    assert 'lr_gamma' not in ProbeParams.model_fields
    with pytest.raises(ValueError):
        ProbeParams(lr_gamma=.25)
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
    assert params['constraint_params']['obj_z_recenter']['start_iter'] == 1
    assert all(c['start_iter'] is None for name, c in params['constraint_params'].items()
               if name != 'obj_z_recenter')
    params['init_params']['probe_source'] = 'custom'
    with pytest.raises(ValueError):
        prepare_parametrized_params(params)


@pytest.mark.parametrize('name,configs', [('AdamW', {'weight_decay': .3}), ('SGD', {'momentum': .9, 'weight_decay': .3})])
def test_vector_updates_match_individual_rates(name, configs):
    overrides = uniform_rate_overrides()
    overrides.update({'C30': {'trainable': False}, 'C10': {'lr': 2.}})
    model, _, _, _ = fixture_model(overrides)
    vector = model.probe_generator.normalized_coefficients
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
    overrides = uniform_rate_overrides()
    overrides.update({'C30': {'trainable': False}, 'C10': {'lr': 2.}})
    model, _, _, _ = fixture_model(overrides)
    optim = create_optimizer({'name': 'AdamW', 'configs': {'weight_decay': .1}}, model.optimizable_params)
    optim.step = torch.compile(optim.step, backend='aot_eager')
    vector = model.probe_generator.normalized_coefficients
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
    torch.testing.assert_close(gpu.normalized_coefficients.grad.cpu(), cpu.normalized_coefficients.grad,
                               atol=1e-6, rtol=1e-4)


def test_high_order_fixed_unless_selected():
    model, values, params, init = fixture_model()
    init['probe_aberrations'] = {'C70': 1000}
    fixed = ParametrizedPtychoModel(values, params, init)
    assert not fixed.probe_generator.trainable_mask[fixed.probe_generator.names.index('C70')]
    params['probe_params']['coefficients']['C70'] = {'lr': 5}
    active = ParametrizedPtychoModel(values, params, init)
    assert active.probe_generator.trainable_mask[active.probe_generator.names.index('C70')]


def test_probe_coefficient_plot_groups_orders_and_requires_probe_selection(tmp_path):
    import matplotlib.pyplot as plt

    model, values, params, init = fixture_model()
    with torch.no_grad():
        model.probe_generator.normalized_coefficients[model.probe_generator.names.index('C30')] = (
            10000 * model.probe_generator.phase_per_angstrom[model.probe_generator.names.index('C30')])
    model.record_probe_coefficients(1)
    assert model.probe_coefficient_iters['niter'] == [0, 1]

    fig = plot_probe_coefficient_curves(model.probe_coefficient_iters)
    panels = [ax for ax in fig.axes if ax.get_visible()]
    assert len(panels) == 5
    assert {line.get_label() for line in panels[0].lines} == {'C10', 'C12a', 'C12b'}
    assert 'C30' in {line.get_label() for line in panels[2].lines}
    plt.close(fig)

    values['pos_pre_affine'] = values['crop_pos'].copy()
    plot_summary(str(tmp_path), model, 1, np.arange(2), values,
                 selected_figs=['probe_r_amp'], show_fig=False, save_fig=True)
    assert (tmp_path / 'summary_probe_coefficients_iter0001.png').is_file()

    plot_summary(str(tmp_path), model, 2, np.arange(2), values,
                 selected_figs=[], show_fig=False, save_fig=True)
    assert not (tmp_path / 'summary_probe_coefficients_iter0002.png').exists()

    params['probe_params']['parametrize'] = False
    pixel_model = create_ptycho_model(SimpleNamespace(init_variables=values, init_params=init),
                                      {'model_params': params})
    plot_summary(str(tmp_path), pixel_model, 3, np.arange(2), values,
                 selected_figs=['probe_r_amp'], show_fig=False, save_fig=True)
    assert not (tmp_path / 'summary_probe_coefficients_iter0003.png').exists()


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
    raw['recon_params'].update(NITER=2, BATCH_SIZE={'size': 2},
                               selected_figs=['probe_r_amp'], save_result=['model', 'optim_state'])
    config = tmp_path / 'params.yaml'
    config.write_text(yaml.safe_dump(raw), encoding='utf8')
    params = load_params(str(config))
    assert params['init_params']['probe_aberrations']['C10'] == .000123456789
    solver = PtyRADSolver(params, device='cpu')
    assert solver.init.init_variables['probe'].shape[0] == 1
    assert params['init_params']['probe_pmode_max'] == 4  # Caller config is untouched.
    solver.reconstruct()
    assert solver.reconstruct_results.probe_coefficient_iters['niter'] == [0, 1, 2]
    assert len(list(tmp_path.rglob('summary_probe_coefficients_iter*.png'))) == 2
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


def test_legacy_coefficients_load_but_optimizer_coordinates_are_rejected(tmp_path):
    model, values, params, init = fixture_model()
    legacy = model.export_parametrized_probe()
    legacy.pop('parameterization')
    legacy['coefficients']['C50'] = -1.85e7
    restored = ParametrizedPtychoModel(values, params, init, state=legacy)
    assert restored.probe_generator.export()['coefficients']['C50'] == pytest.approx(-1.85e7)
    path = tmp_path / 'legacy.hdf5'
    save_dict_to_hdf5({'parametrized_probe': legacy}, str(path))
    with pytest.raises(ValueError, match='incompatible probe coordinates'):
        create_optimizer({'name': 'Adam', 'load_state': str(path)}, restored.optimizable_params)
    params['optimizer_params']['load_state'] = str(path)
    with pytest.raises(ValueError, match='legacy Angstrom'):
        create_ptycho_model(SimpleNamespace(init_variables=values, init_params=init),
                           {'model_params': params})


def test_lbfgs_default_phase_rates_and_single_high_order_coefficient():
    _, values, params, init = fixture_model()
    params['optimizer_params'] = {'name': 'LBFGS', 'configs': {'max_iter': 1}}
    # No order-dependent rates: the default model is accepted by LBFGS.
    ParametrizedPtychoModel(values, params, init)
    params['probe_params']['coefficients'] = {
        name: {'trainable': name == 'C30'} for name in default_coefficients()}
    model = ParametrizedPtychoModel(values, params, init)
    opt = create_optimizer(model.optimizer_params, model.optimizable_params)
    q = model.probe_generator.normalized_coefficients
    i = model.probe_generator.names.index('C30')
    def closure():
        opt.zero_grad()
        loss = (q[i] - 2).square()
        loss.backward()
        return loss
    opt.step(closure)
    assert q[i].item() == pytest.approx(1.)
    assert torch.count_nonzero(q).item() == 1


def test_high_order_phase_fit_has_useful_gradients():
    probe = generator(dtype=torch.float32)
    i = probe.names.index('C50')
    with torch.no_grad():
        probe.normalized_coefficients[i] = .2
        target = probe().detach()
        probe.normalized_coefficients.zero_()
    probe.trainable_mask.zero_()
    probe.trainable_mask[i] = True
    opt = torch.optim.Adam(probe.parameters(), lr=.01)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        loss = (probe() - target).abs().square().sum()
        losses.append(loss.item())
        loss.backward()
        if len(losses) == 1:
            assert probe.normalized_coefficients.grad[i].abs().item() > 1e-5
        opt.step()
    assert losses[-1] < losses[0] * .05
    assert probe.normalized_coefficients[i].item() == pytest.approx(.2, abs=.03)


@pytest.mark.parametrize('distance', [-13.5, 0., 7.25])
@pytest.mark.parametrize('frozen', [False, True])
def test_defocus_shift_matches_paraxial_propagation(distance, frozen):
    gen = generator({'C10': 12.3, 'C12a': 20., 'C70': 200000.}, z_shift=25.)
    index = gen.names.index('C10')
    gen.trainable_mask[index] = not frozen
    parameter = gen.normalized_coefficients
    before = gen.current_coefficients().detach().clone()
    original = gen().detach()
    k = torch.fft.fftfreq(16, d=.2, dtype=torch.float64)
    ky, kx = torch.meshgrid(k, k, indexing='ij')
    transfer = torch.exp(-1j * torch.pi * gen.geometry['wavelength'] * distance *
                         (kx.square() + ky.square()))
    expected = torch.fft.ifft2(torch.fft.fft2(original) * transfer)
    gen.shift_defocus(distance)
    assert gen.normalized_coefficients is parameter
    target = before.clone()
    target[index] += distance
    torch.testing.assert_close(gen.current_coefficients(), target)
    torch.testing.assert_close(gen.fixed_coefficients[index], target[index])
    torch.testing.assert_close(gen(), expected, atol=1e-10, rtol=1e-10)
    gen.shift_defocus(-distance)
    torch.testing.assert_close(gen(), original, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('overrides,start', [({}, 1), ({'C10': {'trainable': False}}, 1),
                                            ({'C10': {'lr': 0.}}, 1), ({}, None)])
def test_recenter_updates_live_probe_and_survives_optimizer(overrides, start, tmp_path):
    from ptyrad.core.constraints import CombinedConstraint, shift_obj_along_z

    _, values, params, init = fixture_model(overrides, start=start)
    phase = np.zeros((1, 5, 20, 20), dtype='float32')
    phase[:, 1] = .3  # CoM=1, desired center=2: move by +1 slice.
    values['obj'] = np.exp(1j * phase).astype('complex64')
    values['slice_thickness'] = 2.5
    model = ParametrizedPtychoModel(values, params, init)
    gen = model.probe_generator
    constraint = CombinedConstraint({'obj_z_recenter': dict(
        start_iter=2, step=2, end_iter=5, thresh=None, scale=1., max_shift=1.)}, device='cpu')
    before = gen.current_coefficients().detach().clone()
    obj_before = torch.polar(model.opt_obja, model.opt_objp).detach()
    optimizer = create_optimizer(model.optimizer_params, model.optimizable_params) if start else None
    with torch.no_grad():
        constraint.apply_obj_z_recenter(model, 1)
        torch.testing.assert_close(gen.current_coefficients(), before)
        constraint.apply_obj_z_recenter(model, 2)
    expected = before.clone()
    expected[gen.names.index('C10')] -= 2.5
    torch.testing.assert_close(gen.current_coefficients(), expected)
    torch.testing.assert_close(torch.polar(model.opt_obja, model.opt_objp),
                               shift_obj_along_z(obj_before, 1.))
    if optimizer:
        # A subsequent zero-gradient step must not restore the old C10.
        gen.normalized_coefficients.grad = torch.zeros_like(gen.normalized_coefficients)
        optimizer.step()
        torch.testing.assert_close(gen.current_coefficients(), expected)
    model.record_probe_coefficients(2)
    assert model.probe_coefficient_iters['coefficients']['C10'][-1] == pytest.approx(-2.5)
    path = tmp_path / 'recentered.hdf5'
    save_dict_to_hdf5({'parametrized_probe': model.export_parametrized_probe()}, str(path))
    restored = ParametrizedPtychoModel(values, params, init,
                                      state=load_ptyrad(str(path))['parametrized_probe'])
    torch.testing.assert_close(restored.get_complex_probe_view(), model.get_complex_probe_view())


@pytest.mark.parametrize('pixel', [False, True])
@pytest.mark.parametrize('source_slice', [1, 2, 3])
def test_recenter_sign_zero_and_pixel_branch(pixel, source_slice):
    from ptyrad.core.constraints import CombinedConstraint
    from ptyrad.core.functional import near_field_evolution_torch
    from ptyrad.core.models.ptycho import PtychoModel

    _, values, params, init = fixture_model()
    phase = np.zeros((1, 5, 20, 20), dtype='float32')
    phase[:, source_slice] = .5
    values['obj'] = np.exp(1j * phase).astype('complex64')
    values['slice_thickness'] = 3.
    model = (PtychoModel(values, params, device='cpu') if pixel else
             ParametrizedPtychoModel(values, params, init))
    before = model.get_complex_probe_view().detach().clone()
    constraint = CombinedConstraint({'obj_z_recenter': dict(
        start_iter=1, step=1, end_iter=None, thresh=None, scale=1., max_shift=1.)}, device='cpu')
    distance = -(2 - source_slice) * 3.
    with torch.no_grad():
        constraint.apply_obj_z_recenter(model, 1)
    if pixel:
        transfer = near_field_evolution_torch(before.shape[-2:], model.dx, distance,
                                              model.lambd, device='cpu')
        torch.testing.assert_close(model.get_complex_probe_view(),
                                   torch.fft.ifft2(torch.fft.fft2(before) * transfer))
    else:
        assert model.export_parametrized_probe()['coefficients']['C10'] == pytest.approx(distance, abs=1e-6)
    # A second call after centering should not add another full-slice correction.
    centered = model.get_complex_probe_view().detach().clone()
    with torch.no_grad():
        constraint.apply_obj_z_recenter(model, 2)
    torch.testing.assert_close(model.get_complex_probe_view(), centered, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('rank_mode,should_log', [('single', True), ('main', True),
                                                ('worker', False), ('ddp_main', True),
                                                ('ddp_worker', False), ('pixel', False)])
def test_final_aberration_yaml_roundtrip(monkeypatch, caplog, rank_mode, should_log):
    import logging
    import yaml
    from ptyrad.params.recon_params import ReconParams
    from ptyrad.solver import reconstruction

    model, values, params, init_params = fixture_model({'C12a': {'trainable': False}})
    init_params['probe_aberrations'] = {'C12a': .123456789, 'C70': 12345.6789}
    init_params['probe_z_shift'] = 25.
    model = ParametrizedPtychoModel(values, params, init_params)
    if rank_mode == 'pixel':
        from ptyrad.core.models.ptycho import PtychoModel
        model = PtychoModel(values, params, device='cpu')
    model.compilation_iters = []
    recon_params = ReconParams().model_dump()
    recon_params.update(NITER=2, SAVE_ITERS=3, convergence_monitor=None)
    def step(*args, **kwargs):
        if hasattr(model, 'probe_generator'):
            model.probe_generator.shift_defocus(-1.25)
        model.iter_times.append(.1)
        return {}
    monkeypatch.setattr(reconstruction, 'recon_step', step)
    monkeypatch.setattr(reconstruction, 'toggle_grad_requires', lambda *args: None)
    monkeypatch.setattr(reconstruction, 'save_results', lambda *args: pytest.fail('Unexpected save'))
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: rank_mode.startswith('ddp'))
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: int(rank_mode == 'ddp_worker'))
    acc = (SimpleNamespace(num_processes=2, is_main_process=rank_mode == 'main')
           if rank_mode in ('main', 'worker') else None)
    with caplog.at_level(logging.INFO, logger=reconstruction.__name__):
        reconstruction.recon_loop(
            model, SimpleNamespace(init_variables=values),
            {'recon_params': recon_params, 'model_params': params},
            SimpleNamespace(step=lambda: None), None, None, None, [], [], '', acc=acc)
    records = [r for r in caplog.records if 'Final parametrized probe aberrations' in r.getMessage()]
    assert len(records) == int(should_log)
    if should_log:
        # Timestamp only prefixes the explanation, not the copyable YAML line.
        output = logging.Formatter('%(asctime)s - %(message)s').format(records[0])
        copied = yaml.safe_load(output.split('\n', 1)[1])
        assert copied['probe_aberrations'] == model.export_parametrized_probe()['coefficients']
        assert copied['probe_aberrations']['C10'] == pytest.approx(-2.5)
        assert 'C70' in copied['probe_aberrations']
        init_params.update(copied)
        restored = ParametrizedPtychoModel(values, params, init_params)
        torch.testing.assert_close(restored.get_complex_probe_view(), model.get_complex_probe_view())
