# Single-mode parametrized STEM probe

Add this to an existing electron reconstruction configuration:

```yaml
model_params:
  probe_params:
    parametrize: true
    coefficients: {}
  update_params:
    probe: {start_iter: 1, lr: 1.0}
```

The default is `parametrize: false`. With the option enabled, all 25 Cartesian
coefficients in the first through fifth order aberration specification are
optimized, including coefficients initially zero. All coefficients are stored in
one trainable tensor of shape `(25,)` (extended when higher orders are included),
with one optimizer parameter group and tensor-valued optimizer state. Names map
to entries in the saved `coefficient_names` list. Initial values are read from
`init_params.probe_aberrations`, using its existing aliases and units. The
learning rate above is an example, in Angstrom per C10 update; the pixel-probe
learning rate may be too small for useful coefficient refinement. By default,
the learning rate of each coefficient is normalized to produce the same maximum
phase change at the aperture edge as C10. For aberration order `n` and probe
convergence semi-angle `alpha` in radians, the multiplier is
`(n + 1) / (2 * alpha ** (n - 1))`. All angular components of the same order
share the multiplier. At 25 mrad, the multipliers for orders 1 through 5 are
1, 60, 3200, 160000, and 7680000.

`coefficients` contains overrides, not initial values or an allowlist:

```yaml
probe_params:
  parametrize: true
  coefficients:
    C30: {trainable: false}
    C10: {lr: 2.0}
    C12b: {lr: 0.0}
```

Use canonical Cartesian names (`C10`, `C12a`, `C12b`, etc.). Other coefficients
continue to optimize. Each inherits the start/end iterations and default learning
rate from `update_params.probe`. An explicit coefficient `lr` uses that value
directly, without the automatic order multiplier. `trainable: false`, zero
coefficient learning rate, or a null probe start iteration freezes the
corresponding coefficients.
Valid higher-order terms supplied in the initial aberrations are fixed unless
explicitly included in `coefficients`.

Per-coefficient learning rates scale entries of the optimizer's actual update,
not the gradients (which Adam would normalize). The normalization balances
phase changes from equal-sized Adam updates; other optimizers may need different
rates because their raw updates depend on gradient magnitude. Frozen entries
remain unchanged even with weight decay or momentum. Use the standard
solver/create_optimizer entry point to install these update hooks. LBFGS requires
a shared effective learning rate, so the default normalization is incompatible
with LBFGS unless all active coefficients have the same explicit `lr`.
Old checkpoints with separate coefficient parameter groups can supply coefficient
values, but their optimizer state cannot be resumed with the new vector layout.

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
