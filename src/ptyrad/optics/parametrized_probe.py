"""Differentiable, single-mode STEM probe with Cartesian aberrations in Angstrom."""

import numpy as np
import torch
from torch import nn

from ptyrad.optics.aberrations import ABERRATION_SPEC, Aberrations
from ptyrad.optics.propagator import near_field_evolution
from ptyrad.params.probe_params import coefficient_order


def default_coefficients():
    return [f"C{n}{m}{suffix}" for n, m in ABERRATION_SPEC
            for suffix in (("",) if m == 0 else ("a", "b"))]


class ParametrizedProbe(nn.Module):
    def __init__(self, *, size, dx, wavelength, conv_angle, intensity,
                 aberrations=None, extra_names=(), z_shift=0.0,
                 device="cpu", dtype=torch.float32):
        super().__init__()
        values = Aberrations(aberrations or {}).get_krivanek_cartesian(decimals=None)
        self.names = sorted(set(default_coefficients()) | set(values) | set(extra_names),
                            key=coefficient_order)
        self.geometry = dict(size=int(size), dx=float(dx), wavelength=float(wavelength),
                             conv_angle=float(conv_angle), intensity=float(intensity),
                             z_shift=float(z_shift or 0))
        if not all(np.isfinite(v) for v in self.geometry.values()):
            raise ValueError("Probe geometry and intensity must be finite")
        if min(size, dx, wavelength, conv_angle, intensity) <= 0:
            raise ValueError("Probe size, sampling, wavelength, angle and intensity must be positive")
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError("Aberration coefficients must be finite")

        # Build fixed grids in double precision before casting to the model dtype.
        k = torch.fft.fftshift(torch.fft.fftfreq(size, d=dx, dtype=torch.float64))
        ky, kx = torch.meshgrid(k, k, indexing="ij")
        ax, ay = kx * wavelength, ky * wavelength
        r2 = ax.square() + ay.square()
        omega = torch.complex(ax, ay)
        basis = []
        for name in self.names:
            n, m, component = coefficient_order(name)
            angular = (omega ** m).imag if component == "b" else (omega ** m).real
            basis.append((2 * torch.pi / wavelength / (n + 1)) *
                         r2.pow((n + 1 - m) // 2) * angular)
        mask = (kx.square() + ky.square()).sqrt() <= conv_angle / 1000 / wavelength
        self.register_buffer("basis", torch.stack(basis).to(device=device, dtype=dtype))
        self.register_buffer("aperture", mask.to(device=device, dtype=dtype))
        self.register_buffer("amplitude", torch.tensor(
            np.sqrt(intensity) * size / np.sqrt(mask.sum().item()), device=device, dtype=dtype))
        # Fixed ASM propagation must match Initializer._probe_z_shift, including phase.
        transfer = near_field_evolution((size, size), dx, z_shift or 0, wavelength)
        complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        # Real buffer supports DDP backends without complex buffer broadcast.
        self.register_buffer("transfer_ri", torch.view_as_real(
            torch.tensor(transfer, device=device, dtype=complex_dtype)))
        self.coefficients = nn.ParameterDict({name: nn.Parameter(torch.tensor(
            [values.get(name, 0.0)], device=device, dtype=dtype)) for name in self.names})

    def from_values(self, values):
        chi = torch.einsum("c,cyx->yx", values, self.basis)
        # exp(-i chi), expressed with real kernels (also avoids CUDA complex-exp JIT).
        pupil = self.aperture * torch.complex(torch.cos(chi), -torch.sin(chi))
        probe = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(pupil))) * self.amplitude
        probe = torch.fft.ifft2(torch.fft.fft2(probe) * torch.view_as_complex(self.transfer_ri))
        return probe.unsqueeze(0)

    def forward(self):
        # Length-one parameters preserve tensor-valued optimizer moments in HDF5.
        return self.from_values(torch.cat([self.coefficients[name] for name in self.names]))

    def export(self):
        return dict(geometry=self.geometry.copy(), coefficients={
            name: value.detach().item() for name, value in self.coefficients.items()})
