"""Opt-in model and integration helpers; the pixel-probe model is unchanged."""

import logging

import numpy as np
import torch

from ptyrad.core.models.ptycho import PtychoModel
from ptyrad.io.load import load_ptyrad
from ptyrad.optics.parametrized_probe import (
    ParametrizedProbe, default_coefficients, phase_normalized_lr_scale,
)
from ptyrad.params.probe_params import ProbeParams

logger = logging.getLogger(__name__)
STATE_KEY = "parametrized_probe"


def enabled(params):
    return params.get("model_params", {}).get("probe_params", {}).get("parametrize", False)


def prepare_parametrized_params(params):
    """Prepare the solver-owned configuration before Initializer and constraints."""
    if not enabled(params):
        return
    ProbeParams.model_validate(params["model_params"]["probe_params"])
    init = params["init_params"]
    if init.get("probe_illum_type", "electron") != "electron":
        raise ValueError("Parametrized probe requires electron illumination")
    source = init.get("probe_source", "simu")
    if source not in ("simu", "PtyRAD"):
        raise ValueError("Parametrized probe requires simulated initialization or a parametrized PtyRAD checkpoint")
    for key in ("probe_permute", "probe_interpolate", "probe_recenter"):
        if init.get(key):
            raise ValueError(f"{key} is incompatible with a parametrized probe")
    if source == "PtyRAD":
        if STATE_KEY not in load_ptyrad(init["probe_params"]):
            raise ValueError("Checkpoint has no parametrized probe coefficients; pixel-probe fitting is not supported")
    tune = params.get("hypertune_params", {})
    if tune.get("if_hypertune") and tune.get("tune_params", {}).get("pmode_max", {}).get("state"):
        raise ValueError("Cannot tune pmode_max with a single-mode parametrized probe")
    if init.get("probe_pmode_max", 1) != 1:
        logger.warning("Parametrized probe: forcing probe_pmode_max=1")
    init.update(probe_pmode_max=1, probe_pmode_init_pows=[1.0])
    # Older/direct dictionary callers may also supply the simulation mode count.
    if "pmodes" in init:
        init["pmodes"] = 1
    for name in ("probe_mask_k", "probe_mask_r", "obj_z_recenter", "ortho_pmode", "fix_probe_int"):
        config = params.get("constraint_params", {}).get(name)
        if config is not None and config.get("start_iter") is not None:
            logger.warning("Parametrized probe: disabling %s (fixed aperture, intensity and single mode)", name)
            config["start_iter"] = None


class ParametrizedPtychoModel(PtychoModel):
    def __init__(self, init_variables, model_params, init_params, device="cpu", state=None):
        # Base initialization uses the original getter until the generator is installed.
        super().__init__(init_variables, model_params, device=device)
        config = ProbeParams.model_validate(model_params.get("probe_params", {})).model_dump()
        overrides = config["coefficients"]
        geometry = dict(size=int(init_variables["probe"].shape[-1]),
                        dx=float(init_variables["dx"]), wavelength=float(init_variables["lambd"]),
                        conv_angle=float(init_params["probe_conv_angle"]),
                        intensity=float(np.square(np.abs(init_variables["probe"])).sum()),
                        z_shift=float(init_params.get("probe_z_shift") or 0))
        if init_variables["probe"].shape != (1, geometry["size"], geometry["size"]):
            raise ValueError("Parametrized probe requires one square probe mode")
        aberrations = init_params.get("probe_aberrations", {})
        if state is not None:
            for key in ("size", "dx", "wavelength", "conv_angle"):
                if not np.isclose(geometry[key], state["geometry"][key], rtol=1e-6, atol=0):
                    raise ValueError(f"Parametrized checkpoint geometry mismatch: {key}")
            if geometry["z_shift"] not in (0, state["geometry"]["z_shift"]):
                raise ValueError("Cannot change probe_z_shift on parametrized resume")
            geometry = dict(state["geometry"])
            geometry["size"] = int(geometry["size"])
            aberrations = state["coefficients"]
        self.probe_generator = ParametrizedProbe(**geometry, aberrations=aberrations,
                                                 extra_names=overrides, device=device)
        self.probe_config = config
        initial_coefficients = self.probe_generator.current_coefficients().detach().cpu().tolist()
        self.probe_coefficient_iters = {
            'niter': [0],
            'coefficients': {name: [value] for name, value in
                             zip(self.probe_generator.names, initial_coefficients)},
        }
        self.probe_update = dict(model_params['update_params']['probe'])
        # Keep the historical probe dictionary entry as an inert buffer for exporters.
        # All consumers get the current complex probe through get_complex_probe_view().
        del self.opt_probe
        self.register_buffer("opt_probe", torch.view_as_real(self.probe_generator().detach()).clone())
        self.optimizable_tensors["probe"] = self.opt_probe
        self.lr_params["probe"] = 0.0
        self.start_iter["probe"] = None
        self.end_iter["probe"] = None
        update = model_params["update_params"]["probe"]
        defaults = set(default_coefficients())
        rates = []
        configured_rates = []
        for name in self.probe_generator.names:
            override = overrides.get(name, {})
            trainable = override.get("trainable", True) and (name in defaults or name in overrides)
            explicit_lr = override.get("lr")
            configured_lr = update["lr"] if explicit_lr is None else explicit_lr
            active = trainable and configured_lr > 0 and update.get("start_iter") is not None
            configured_rates.append(configured_lr if active else 0.0)
            scale = (phase_normalized_lr_scale(name, geometry["conv_angle"])
                     if explicit_lr is None else 1.0)
            rates.append(configured_lr * scale if active else 0.0)
        key = "probe_coefficients"
        tensor = self.probe_generator.coefficients
        self.probe_generator.trainable_mask.copy_(
            torch.tensor([rate > 0 for rate in rates], device=tensor.device))
        # Keep the configured probe LR as the optimizer LR. The step hook applies
        # potentially much larger high-order multipliers after optimizer.step().
        group_lr = max(configured_rates)
        self.optimizable_tensors[key] = tensor
        self.lr_params[key] = group_lr
        self.start_iter[key] = update.get("start_iter") if group_lr > 0 else None
        self.end_iter[key] = update.get("end_iter") if group_lr > 0 else None
        self.probe_int_sum = self.get_complex_probe_view().detach().abs().square().sum()
        self.create_optimizable_params_dict(self.lr_params)
        for group in self.optimizable_params:
            if group['params'][0] is tensor:
                group['probe_update_scale'] = [rate / group_lr for rate in rates]
        if self.optimizer_params['name'] == 'LBFGS' and len(set(r for r in rates if r > 0)) > 1:
            raise ValueError("LBFGS requires one shared learning rate for the probe coefficient tensor")
        self.init_compilation_iters()

    def get_complex_probe_view(self):
        if hasattr(self, "probe_generator"):
            return self.probe_generator()
        return super().get_complex_probe_view()

    def record_probe_coefficients(self, niter):
        values = self.probe_generator.current_coefficients().detach().cpu().tolist()
        self.probe_coefficient_iters['niter'].append(niter)
        for name, value in zip(self.probe_generator.names, values):
            self.probe_coefficient_iters['coefficients'][name].append(value)

    def export_parametrized_probe(self):
        state = self.probe_generator.export()
        state["config"] = self.probe_config
        state["update_params"] = self.probe_update
        state["parameter_groups"] = [name for name, lr in self.lr_params.items() if lr != 0]
        state["coefficient_names"] = self.probe_generator.names
        return state

    def print_model_summary(self):
        if not any(t.requires_grad for t in self.optimizable_tensors.values()):
            logger.info("No active parameters at initialization; check update schedules")
            return
        super().print_model_summary()


def create_ptycho_model(init, params, device="cpu"):
    if not enabled(params):
        return PtychoModel(init.init_variables, params["model_params"], device=device)
    state = None
    if init.init_params.get("probe_source") == "PtyRAD":
        state = load_ptyrad(init.init_params["probe_params"]).get(STATE_KEY)
        if state is None:
            raise ValueError("Checkpoint has no parametrized probe coefficients")
    model = ParametrizedPtychoModel(init.init_variables, params["model_params"],
                                  init.init_params, device=device, state=state)
    # Keep initial plots identical to the actual model, also after checkpoint restoration.
    init.init_variables["probe"] = model.get_complex_probe_view().detach().cpu().numpy()
    optim_path = model.optimizer_params.get("load_state")
    if optim_path:
        saved = load_ptyrad(optim_path).get(STATE_KEY)
        if saved is None or list(saved["parameter_groups"]) != model.export_parametrized_probe()["parameter_groups"]:
            raise ValueError("Optimizer checkpoint has incompatible parametrized probe parameter groups")
        if list(saved.get('coefficient_names', [])) != model.probe_generator.names:
            raise ValueError("Optimizer checkpoint has incompatible probe coefficient ordering")
        if saved['config'] != model.probe_config or saved['update_params'] != model.probe_update:
            raise ValueError("Optimizer checkpoint has incompatible probe training settings")
        for key, value in model.probe_generator.geometry.items():
            if not np.isclose(value, saved['geometry'][key], rtol=1e-6, atol=0):
                raise ValueError(f"Optimizer checkpoint probe geometry mismatch: {key}")
    return model
