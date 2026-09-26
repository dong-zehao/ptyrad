"""Elementwise update scaling for a single probe coefficient parameter."""

import torch


def configure_probe_updates(optimizer):
    """Scale actual updates, rather than gradients which Adam would normalize away.

    A single parameter group and moment tensor are retained. Frozen entries are
    restored even with momentum or weight decay. The shared scheduler scales all
    coefficient rates while preserving their ratios.
    """
    groups = [group for group in optimizer.param_groups if 'probe_update_scale' in group]
    if not groups:
        return optimizer
    scales = [torch.as_tensor(group['probe_update_scale'], device=group['params'][0].device,
                              dtype=group['params'][0].dtype) for group in groups]
    before = []

    @torch.no_grad()
    def snapshot(optim, args, kwargs):
        before[:] = [g['params'][0].detach().clone() for g in groups]

    @torch.no_grad()
    def scale_updates(optim, args, kwargs):
        for group, previous, scale in zip(groups, before, scales):
            parameter = group['params'][0]
            parameter.copy_(torch.where(scale == 0, previous,
                                        previous + (parameter - previous) * scale))
        before.clear()

    optimizer.register_step_pre_hook(snapshot)
    optimizer.register_step_post_hook(scale_updates)
    return optimizer
