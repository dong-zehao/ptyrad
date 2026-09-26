# Single-mode parametrized STEM probe

Add this to an existing electron reconstruction configuration:

```yaml
model_params:
  probe_params:
    parametrize: true
    coefficients: {}
  update_params:
    probe: {start_iter: 1, lr: 0.001}
```

The default is `parametrize: false`. With the option enabled, all 25 Cartesian
coefficients in the first through fifth order aberration specification are
optimized, including coefficients initially zero. The trainable tensor
`normalized_coefficients` has shape `(25,)` (extended for higher orders).
For each physical Cartesian coefficient `C_nm` in Angstroms, it stores

```
q_nm = C_nm * (2*pi / wavelength) * alpha**(n+1) / (n+1)
```

where wavelength is in Angstroms and alpha is the convergence semi-angle in
radians. Thus `q_nm` is the angular-component amplitude of the aberration phase
at the aperture edge, in radians. The forward pass converts `q_nm` back to
Angstroms before generating the probe. Initialization, plots, and exported
`parametrized_probe.coefficients` remain in Angstroms; the optimizer tensor and
its moments use phase coordinates. This balances gradient scales across orders
and avoids taking tiny Angstrom steps on large physical coefficients.

Both `update_params.probe.lr` and coefficient-specific `lr` now act on phase
coordinates, not Angstroms. The example above uses 0.001; retune old learning
rates rather than copying their numerical values blindly. `lr_gamma` has been
removed; remove it from existing configurations (validation rejects it).

`coefficients` contains overrides, not initial values or an allowlist:

```yaml
probe_params:
  parametrize: true
  coefficients:
    C30: {trainable: false}
    C10: {lr: 0.002}
    C12b: {lr: 0.0}
```

Use canonical Cartesian names (`C10`, `C12a`, `C12b`, etc.). Other coefficients
continue to optimize. Each inherits the start/end iterations and default learning
rate from `update_params.probe`. An explicit coefficient `lr` overrides the phase-coordinate learning rate. `trainable: false`, zero
coefficient learning rate, or a null probe start iteration freezes the
corresponding coefficients.
Valid higher-order terms supplied in the initial aberrations are fixed unless
explicitly included in `coefficients`.

Per-coefficient learning rates scale entries of the optimizer's actual update
in phase coordinates, not the gradients (which Adam would normalize).
Frozen entries remain unchanged even with weight decay or momentum. Use the
standard solver/create_optimizer entry point to install these override hooks.
LBFGS requires a shared learning rate for active probe coefficients; the default
configuration now satisfies this requirement. As in the pixel model, LBFGS uses
a single global learning rate across all model parameters.

Legacy checkpoints can supply physical coefficient values, which are converted
to phase coordinates on load. Their optimizer moments cannot be resumed: omit
`optimizer_params.load_state` and start a fresh optimizer (and scheduler).
New checkpoints identify the coordinate system as `aperture_phase_radians_v1`
and support optimizer-state restoration with matching geometry and settings.

When `selected_figs` includes `probe_r_amp`, `probe_k_amp`, `probe_k_phase`, or
`all`, each save interval also writes `summary_probe_coefficients_iterNNNN.png`.
The figure tracks every coefficient from initialization through each iteration,
with one subplot per aberration order.

The probe has one mode, a fixed circular aperture, fixed convergence angle, and
fixed total intensity determined by the existing initialization normalization.
`probe_z_shift` is a fixed angular-spectrum propagation after probe formation.
Probe masks, mode orthogonalization, fixed-intensity projection, and object
z-recentering constraints are disabled with a log message. The remaining
constraints and losses are unchanged. Probe permutation, interpolation,
recentering, external pixel-probe fitting, X-ray illumination, and mode-count
hyperparameter searches are unsupported in this mode.
`init_params.probe_interpolate: null` disables interpolation and is supported.

Saved model HDF5 files retain the usual complex probe and include a
`parametrized_probe` group with current coefficients, geometry, training overrides,
and parameter-group ordering. Set `init_params.probe_source: PtyRAD` and
`init_params.probe_params` to such a model file to resume. Keep matching grid,
wavelength, aperture, and training configuration; optimizer state can be restored
through the existing `optimizer_params.load_state`. Pixel-only checkpoints cannot
initialize a parametrized reconstruction. Coefficients and fixed intensity from
the checkpoint take precedence over initialization values on resume.
