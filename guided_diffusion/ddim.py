import logging
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_ckpt
from tqdm import tqdm
import torch.nn.functional as F

from utils.logger import logging_info
from .gaussian_diffusion import _extract_into_tensor
from .new_scheduler import ddim_timesteps, ddim_repaint_timesteps
from .respace import SpacedDiffusion
from torch.optim.lr_scheduler import ReduceLROnPlateau



def noise_like(shape, device, repeat=False):
    def repeat_noise():
        return torch.randn((1, *shape[1:]), device=device).repeat(
            shape[0], *((1,) * (len(shape) - 1))
        )

    def noise():
        return torch.randn(shape, device=device)

    return repeat_noise() if repeat else noise()


class DDIMSampler(SpacedDiffusion):
    def __init__(self, use_timesteps, conf=None, **kwargs):
        super().__init__(
            use_timesteps=use_timesteps,
            conf=conf,
            **kwargs,
        )
        self.ddim_sigma = conf.get("ddim.ddim_sigma", 0.0)

    def _get_et(self, model_fn, x, t, model_kwargs):
        model_fn = self._wrap_model(model_fn)
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model_fn(x, self._scale_timesteps(t), **model_kwargs)
        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, _ = torch.split(model_output, C, dim=1)
        return model_output

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(
                self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_sample(
        self,
        model_fn,
        x,
        t,
        prev_t,
        model_kwargs,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        **kwargs,
    ):
        B, C = x.shape[:2]
        assert t.shape == (B,)
        with torch.no_grad():
            alpha_t = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
            alpha_prev = _extract_into_tensor(
                self.alphas_cumprod, prev_t, x.shape)
            sigmas = (
                self.ddim_sigma
                * torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
                * torch.sqrt((1 - alpha_t / alpha_prev))
            )

            def process_xstart(_x):
                if denoised_fn is not None:
                    _x = denoised_fn(_x)
                if clip_denoised:
                    return _x.clamp(-1, 1)
                return _x

            e_t = self._get_et(model_fn, x, t, model_kwargs)
            pred_x0 = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=e_t))

            mean_pred = (
                pred_x0 * torch.sqrt(alpha_prev)
                + torch.sqrt(1 - alpha_prev - sigmas**2) * e_t
            )
            noise = noise_like(x.shape, x.device, repeat=False)

            nonzero_mask = (t != 0).float().view(-1, *
                                                 ([1] * (len(x.shape) - 1)))
            x_prev = mean_pred + noise * sigmas * nonzero_mask

        return {
            "x_prev": x_prev,
            "pred_x0": pred_x0,
        }

    def q_sample_middle(self, x, cur_t, tar_t, no_noise=False):
        assert cur_t <= tar_t
        device = x.device
        while cur_t < tar_t:
            if no_noise:
                noise = torch.zeros_like(x)
            else:
                noise = torch.randn_like(x)
            _cur_t = torch.tensor(cur_t, device=device)
            beta = _extract_into_tensor(self.betas, _cur_t, x.shape)
            x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * noise
            cur_t += 1
        return x

    def q_sample(self, x_start, t, no_noise=False):
        if no_noise:
            noise = torch.zeros_like(x_start)
        else:
            noise = torch.randn_like(x_start)

        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod,
                                 t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def x_forward_sample(self, x0, forward_method="from_0", no_noise=False):
        x_forward = [self.q_sample(x0, torch.tensor(0, device=x0.device))]
        if forward_method == "from_middle":
            for _step in range(0, len(self.timestep_map) - 1):
                x_forward.append(
                    self.q_sample_middle(
                        x=x_forward[-1][0].unsqueeze(0),
                        cur_t=_step,
                        tar_t=_step + 1,
                        no_noise=no_noise,
                    )
                )
        elif forward_method == "from_0":
            for _step in range(1, len(self.timestep_map)):
                x_forward.append(
                    self.q_sample(
                        x_start=x0[0].unsqueeze(0),
                        t=torch.tensor(_step, device=x0.device),
                        no_noise=no_noise,
                    )
                )
        return x_forward

    def p_sample_loop(
        self,
        model_fn,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        return_all=False,
        conf=None,
        sample_dir="",
        **kwargs,
    ):
        if device is None:
            device = next(model_fn.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(shape, device=device)

        assert conf["ddim.schedule_params"] is not None
        steps = ddim_timesteps(**conf["ddim.schedule_params"])
        time_pairs = list(zip(steps[:-1], steps[1:]))

        x0 = model_kwargs["gt"]
        x_forwards = self.x_forward_sample(x0)
        mask = model_kwargs["masks"]

        x_t = img
        import os
        from utils import normalize_image, save_grid

        for cur_t, prev_t in tqdm(time_pairs):
            # replace surrounding
            x_t = x_forwards[cur_t] * mask + (1.0 - mask) * x_t
            cur_t = torch.tensor([cur_t] * shape[0], device=device)
            prev_t = torch.tensor([prev_t] * shape[0], device=device)

            output = self.p_sample(
                model_fn,
                x=x_t,
                t=cur_t,
                prev_t=prev_t,
                model_kwargs=model_kwargs,
                conf=conf,
                pred_xstart=None,
            )
            x_t = output["x_prev"]

            if conf["debug"]:
                from utils import normalize_image, save_grid

                os.makedirs(os.path.join(sample_dir, "middles"), exist_ok=True)
                save_grid(
                    normalize_image(x_t),
                    os.path.join(sample_dir, "middles",
                                 f"mid-{prev_t[0].item()}.png"),
                )
                save_grid(
                    normalize_image(output["pred_x0"]),
                    os.path.join(sample_dir, "middles",
                                 f"pred-{prev_t[0].item()}.png"),
                )

        x_t = x_t.clamp(-1.0, 1.0)
        return {
            "sample": x_t,
        }


# implemenet


class G_DDIMSampler(DDIMSampler):
    def __init__(self, use_timesteps, conf=None, **kwargs):
        super().__init__(
            use_timesteps=use_timesteps,
            conf=conf,
            **kwargs,
        )
        self.mid_interval_num = int(conf.get("mid_interval_num", 1))
        self.steps = ddim_timesteps(**conf["ddim.schedule_params"])
        self.mode = conf.get("mode", "denoise")
        self.denoise_step = conf.get("ddim.denoise_step", 5)


    def p_sample(
            self,
            model_fn,
            x,
            t,
            prev_t,
            model_kwargs,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            **kwargs,
    ):
        def reg_fn(_origin_xt, _xt):
            ret = torch.sum((_origin_xt - _xt) ** 2)
            return ret

        def process_xstart(_x):
            if denoised_fn is not None:
                _x = denoised_fn(_x)
            if clip_denoised:
                return _x.clamp(-1.0, 1.0)
            return _x

        def get_et(_x, _t):
            res = self._get_et(model_fn, _x, _t, model_kwargs)
            return res


        def get_predx0(_x, _t, _et, interval_num=1):
            if interval_num == 1:
                return process_xstart(self._predict_xstart_from_eps(_x, _t, _et))

        def get_update(
            _x,
            cur_t,
            _prev_t,
            _et=None,
            _pred_x0=None,
        ):
            if _et is None:
                _et = get_et(_x=_x, _t=cur_t)
            if _pred_x0 is None:
                _pred_x0 = get_predx0(_x, cur_t, _et, interval_num=1)

            alpha_t = _extract_into_tensor(self.alphas_cumprod, cur_t, _x.shape)
            alpha_prev = _extract_into_tensor(
                self.alphas_cumprod, _prev_t, _x.shape)
            sigmas = (
                self.ddim_sigma
                * torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
                * torch.sqrt((1 - alpha_t / alpha_prev))
            )
            mean_pred = (
                _pred_x0 * torch.sqrt(alpha_prev)
                + torch.sqrt(1 - alpha_prev - sigmas**2) * _et  # dir_xt
            )
            noise = noise_like(_x.shape, _x.device, repeat=False)
            nonzero_mask = (cur_t != 0).float().view(-1,
                                                     *([1] * (len(_x.shape) - 1)))
            _x_prev = mean_pred + noise * sigmas * nonzero_mask
            return _x_prev


        def get_xt_from_x0(_x0, _t):
            alpha_t = _extract_into_tensor(self.alphas_cumprod, _t, _x0.shape)
            x_t = torch.sqrt(alpha_t) * _x0 + torch.sqrt(1.0 - alpha_t) * torch.randn(_x0.shape, device=_x0.device)
            return x_t.detach()


        with torch.no_grad(): # 节省显存，防止爆显存
            B, C = x.shape[:2]
            assert t.shape == (B,)
            x0 = model_kwargs["gt"]
            x = x.detach()
            origin_x = x.clone().detach()
            before_denoise_mse = reg_fn(x0, x).item()
            logging_info(f"before_denoise_mse: {before_denoise_mse:.3f}")

            e_t = get_et(x, _t=t).detach()
            pred_x0 = get_predx0(x, _t=t, _et=e_t, interval_num=self.mid_interval_num).detach() # 关键，必须更新

            x_prev = get_update(
                x,
                t,
                prev_t,
                e_t,
                _pred_x0=pred_x0 if self.mid_interval_num == 1 else None,
            )

            new_mse = reg_fn(x0, x_prev).item()
            logging_info(f"MSE Change: %.3lf -> %.3lf" % (before_denoise_mse, new_mse))
            new_reg = reg_fn(origin_x, x_prev).item()
            logging_info("Reg Change: %.3lf -> %.3lf" % (0, new_reg))

            del new_reg, before_denoise_mse, origin_x,
            torch.cuda.empty_cache()

            return {"x": x, "x_prev": x_prev, "pred_x0": pred_x0, "loss": new_mse}

    def p_sample_loop(
            self,
            model_fn,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=True,
            return_all=False,
            conf=None,
            sample_dir="",
            **kwargs,
    ):
        if device is None:
            device = next(model_fn.parameters()).device
        assert isinstance(shape, (tuple, list))

        logging_info(f"time_steps: {self.steps}")
        time_pairs = list(zip(self.steps[:-1], self.steps[1:]))

        # set up hyper paramer for this run
        if self.mode == "denoise":
            x_t = model_kwargs["gt"]
        loss = None

        status = None
        if self.mode == "denoise":
            for cur_t, prev_t in tqdm(time_pairs[-self.denoise_step:]):

                logging_info(
                    f"cur_t: {cur_t}, next_t: {prev_t}"
                )
                if cur_t > prev_t:  # denoise
                    status = "reverse"
                    cur_t = torch.tensor([cur_t] * shape[0], device=device)
                    prev_t = torch.tensor([prev_t] * shape[0], device=device)
                    output = self.p_sample(
                        model_fn,
                        x=x_t,
                        t=cur_t,
                        prev_t=prev_t,
                        model_kwargs=model_kwargs,
                        pred_xstart=None,
                        cond_fn=cond_fn,
                    )
                    x_t = output["x_prev"]
                    loss = output["loss"]

                    if conf["debug"]:
                        from utils import normalize_image, save_grid

                        os.makedirs(os.path.join(sample_dir, "middles"), exist_ok=True)

                        save_grid(
                            normalize_image(output["x"]),  # 保存当前步优化后的 x_t
                            os.path.join(
                                sample_dir, "middles", f"mid-{cur_t[0].item()}.png"
                            ),
                        )
                        # save_grid(
                        #     normalize_image(output["x_prev"]),
                        #     os.path.join(
                        #         sample_dir, "middles", f"mid_next-{prev_t[0].item()}.png"
                        #     )
                        # )

                        save_grid(
                            normalize_image(output["pred_x0"]),  # 根据当前步优化后的x_t预测的x0
                            os.path.join(
                                sample_dir, "middles", f"{cur_t[0].item()}-pred.png"
                            ),
                        )

                        if output["comb_x0"] is not None:
                            save_grid(
                                normalize_image(output["comb_x0"]),
                                os.path.join(
                                    sample_dir, "middles", f"{cur_t[0].item()}-zcomb.png"
                                )
                            )

        x_t = x_t.clamp(-1.0, 1.0)  # normalize
        return {"sample": x_t, "loss": loss}
