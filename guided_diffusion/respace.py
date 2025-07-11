# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

import numpy as np
import torch as th

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
        section_counts = [int(x) for x in section_counts.split(",")]
    if isinstance(section_counts, int):
        section_counts = [section_counts]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []

    if len(section_counts) == 1 and section_counts[0] > num_timesteps:
        return set(np.linspace(start=0, stop=num_timesteps, num=section_counts[0]))

    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            # frac_stride = (size - 1) / (section_count - 1)
            frac_stride = size // section_count
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx)) # round ��������
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    if num_timesteps-1 not in all_steps:
        all_steps.append(num_timesteps-1)
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, conf=None, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.original_num_steps = len(kwargs["betas"])
        self.conf = conf

        base_diffusion = GaussianDiffusion(
            conf=conf, **kwargs
        )  # pylint: disable=missing-kwoa

        if conf.respace_interpolate:
            raise NotImplementedError
        else:
            self.timestep_map = []
            new_betas = []
            last_alpha_cumprod = 1.0
            for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
                if i in self.use_timesteps:
                    new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                    last_alpha_cumprod = alpha_cumprod
                    self.timestep_map.append(i)

        kwargs["betas"] = np.array(new_betas)

        super().__init__(conf=conf, **kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    # def training_losses(
    #         self, model, *args, **kwargs
    # ):  # pylint: disable=signature-differs
    #     return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    # def condition_score(self, cond_fn, *args, **kwargs):
    #     return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model,
            self.timestep_map,
            self.rescale_timesteps,
            self.original_num_steps,
            self.conf,
        )

    @staticmethod
    def _scale_timesteps(t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(
        self, model, timestep_map, rescale_timesteps, original_num_steps, conf
    ):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps
        self.conf = conf

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(  # pylint: disable=not-callable
            self.timestep_map, device=ts.device, dtype=ts.dtype
        )
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            raise NotImplementedError()
            # new_ts = self.do_rescale_timesteps(new_ts)

        if self.conf.respace_interpolate:
            new_ts = new_ts.float() * (
                (self.conf.diffusion_steps - 1)
                / (float(self.conf.timestep_respacing) - 1.0)
            )

        return self.model(x, new_ts, **kwargs)

    def do_rescale_timesteps(self, new_ts):
        new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return new_ts
