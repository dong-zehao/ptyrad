"""Differentiable second-order consistency penalties for an electron probe pupil."""

import torch


def interior_aperture_mask(shape, dx, wavelength, conv_angle_mrad, fraction, device):
    """Return the centered pupil interior, excluding the physical aperture edge."""
    ny, nx = shape
    fy = torch.fft.fftshift(torch.fft.fftfreq(ny, d=dx, device=device))
    fx = torch.fft.fftshift(torch.fft.fftfreq(nx, d=dx, device=device))
    ky, kx = torch.meshgrid(fy, fx, indexing="ij")
    mask = torch.hypot(ky, kx) * wavelength * 1000 <= fraction * conv_angle_mrad

    if ny < 3 or nx < 3 or not _triple_mask(mask, 0).any() or not _triple_mask(mask, 1).any():
        raise ValueError("loss_probe_reg aperture contains no three-pixel triples in both directions")
    return mask


def _triple_mask(mask, axis):
    if axis == 0:
        return mask[:-2, :] & mask[1:-1, :] & mask[2:, :]
    return mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:]


def _triples(values, axis):
    if axis == 0:
        return values[..., :-2, :], values[..., 1:-1, :], values[..., 2:, :]
    return values[..., :, :-2], values[..., :, 1:-1], values[..., :, 2:]


def _centered_pupil(probe):
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(probe, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )


def _normalize_pupil(pupil, mask):
    tiny = torch.finfo(pupil.real.dtype).tiny
    power = (pupil.abs().square() * mask).sum()
    return pupil / power.clamp_min(tiny).sqrt()


def product_curvature(probe, mask):
    """Normalized |Q_i Q_k - Q_j^2|^2 on the stored dominant probe mode."""
    pupil = _normalize_pupil(_centered_pupil(probe[0]), mask)
    numerator = pupil.real.new_zeros(())
    denominator = pupil.real.new_zeros(())
    for axis in (0, 1):
        left, center, right = _triples(pupil, axis)
        valid = _triple_mask(mask, axis)
        endpoints = left * right
        midpoint = center.square()
        numerator = numerator + ((endpoints - midpoint).abs().square() * valid).sum()
        denominator = denominator + ((endpoints.abs().square() + midpoint.abs().square()) * valid).sum()
    return numerator / denominator.clamp_min(torch.finfo(pupil.real.dtype).tiny)


def mixed_curvature(probe, mask):
    """Unitary-mode-invariant local mutual-coherence curvature."""
    pupil = _normalize_pupil(_centered_pupil(probe), mask)
    intensity = pupil.abs().square().sum(dim=0)
    numerator = pupil.real.new_zeros(())
    denominator = pupil.real.new_zeros(())
    for axis in (0, 1):
        left, center, right = _triples(pupil, axis)
        si, sj, sk = _triples(intensity, axis)
        valid = _triple_mask(mask, axis)
        cij = (left * center.conj()).sum(dim=0)
        cjk = (center * right.conj()).sum(dim=0)
        weight = sj * (si * sk).clamp_min(torch.finfo(sj.dtype).tiny).sqrt()
        numerator = numerator + ((weight - (cij * cjk.conj()).real) * valid).sum()
        denominator = denominator + (weight * valid).sum()
    return numerator / denominator.clamp_min(torch.finfo(pupil.real.dtype).tiny)
