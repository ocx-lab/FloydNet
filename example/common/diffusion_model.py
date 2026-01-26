# Copyright 2025 Beijing Academy of Artificial Intelligence (BAAI)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn as nn


class SigmoidNoiseScheduler:
    def __init__(
        self,
        noise_schedule_factor_a=-4,
        noise_schedule_factor_b=4,
    ):
        self.noise_schedule_factor_a = noise_schedule_factor_a
        self.noise_schedule_factor_b = noise_schedule_factor_b

    def compute_alpha_bar(
        self,
        num_diffusion_steps,
    ):
        def logistic_sigmoid(x):
            return 1 / (1 + math.exp(-x))

        range_start = self.noise_schedule_factor_a
        range_end = self.noise_schedule_factor_b
        value_start = logistic_sigmoid(range_start)
        value_end = logistic_sigmoid(range_end)

        alpha_bar = [
            (value_end - logistic_sigmoid(t / num_diffusion_steps * (range_end - range_start) + range_start)) / (value_end - value_start)
            for t in range(num_diffusion_steps + 1)
        ]
        beta = [0.] + [1 - (alpha_bar[t] / alpha_bar[t - 1]) for t in range(1, num_diffusion_steps + 1)]

        return alpha_bar, beta


class DDPM(nn.Module):
    def __init__(
        self,
        num_diffusion_steps=10000,
        sampling_step_size=200,
        eta_sampling=1.0,
        ns_reweight_clamp: float = 1.0,
    ):
        super().__init__()
        self.num_diffusion_steps = num_diffusion_steps
        self.sampling_step_size = sampling_step_size
        self.eta_sampling = eta_sampling
        self.ns_reweight_clamp = ns_reweight_clamp
        self.ns_density_fn = lambda x: 1 + 8 * (x ** 8)

        noise_scheduler = SigmoidNoiseScheduler()
        alpha_bar, self.beta = noise_scheduler.compute_alpha_bar(self.num_diffusion_steps)

        self.register_buffer("alpha_bar_", torch.tensor(alpha_bar, dtype=torch.float32), persistent=False)

    def sqrt(self, x: torch.Tensor):
        return torch.sqrt(x.clamp(min=0.))

    def alpha_bar(self, t, shape):
        alpha_bar_t = self.alpha_bar_[t]
        while alpha_bar_t.ndim < len(shape):
            alpha_bar_t = alpha_bar_t[..., None]
        return alpha_bar_t.expand(shape)

    def get_t(self, device, B):
        return self.noise_schedule_t_sample(device, B)

    @torch.no_grad()
    def q_sample(self, graph_batch, t=None):
        x0 = graph_batch.adj_label
        sh = x0.shape
        B = sh[0]
        device = x0.device
        t = self.get_t(device, B)
        alpha_bar_t = self.alpha_bar(t, (1,))

        graph_batch["t"] = t
        graph_batch["alpha_bar_t"] = alpha_bar_t

        noise = torch.randn(x0.shape, dtype=torch.float32, device=device)
        xt = self.sqrt(alpha_bar_t) * x0 + self.sqrt(1 - alpha_bar_t) * noise
        graph_batch.xt = xt

        return graph_batch
    
    def noise_schedule_t_sample(self, device, batch_size):
        t_weights = torch.arange(self.num_diffusion_steps + 1, dtype=torch.float64, device=device) / self.num_diffusion_steps
        t_weights = self.ns_density_fn(t_weights)
        t_weights = t_weights.cumsum(dim=-1)
        t = torch.searchsorted(t_weights, torch.rand((batch_size,), dtype=torch.float64, device=device) * t_weights[-1])
        return t
    
    def noise_schedule_reweight(self, loss, t):
        weight = 1.0 / max(max(1.0 - self.alpha_bar_[t].item(), 0.0) ** 0.5, self.ns_reweight_clamp)
        loss = loss * weight
        return loss

    def make_pure_noise(self, graph_batch):
        x0 = graph_batch.adj_label
        graph_batch.xt = torch.randn(x0.shape, dtype=torch.float32, device=x0.device)

        return graph_batch

    def get_pred_noise_from_pred_x0(self, xt, t, pred_x0):
        alpha_bar_t = self.alpha_bar(t, xt.shape)
        pred_noise = (xt - self.sqrt(alpha_bar_t) * pred_x0) \
                        / self.sqrt(1 - alpha_bar_t)
        return pred_noise

    def get_pred_x0_from_pred_noise(self, xt, t, pred_noise):
        alpha_bar_t = self.alpha_bar(t, xt.shape)
        pred_x0 = (xt - self.sqrt(1 - alpha_bar_t) * pred_noise) \
                    / self.sqrt(alpha_bar_t)
        return pred_x0

    def reverse_diffusion(self, xt, t, t_new, eta, pred_x0, pred_noise, graph_batch):
        assert pred_x0 is not None or pred_noise is not None

        if pred_x0 is None:
            pred_x0 = self.get_pred_x0_from_pred_noise(xt, t, pred_noise)

        if pred_noise is None:
            pred_noise = self.get_pred_noise_from_pred_x0(xt, t, pred_x0)

        t_new = t_new.clamp(min=0)
        alpha_bar_t = self.alpha_bar(t, xt.shape)
        alpha_bar_t_new = self.alpha_bar(t_new, xt.shape)

        sigma_t = eta * self.sqrt((1 - alpha_bar_t_new) / 
                                  (1 - alpha_bar_t) * 
                                  (1 - (alpha_bar_t / alpha_bar_t_new)))
        new_noise = torch.randn_like(pred_x0)
        a = self.sqrt(alpha_bar_t_new) * pred_x0
        b = self.sqrt(1 - alpha_bar_t_new - (sigma_t ** 2)) * pred_noise
        c = sigma_t * new_noise
        xt_new = a + b + c
        return xt_new


    def generate(self, graph_batch, model):
        t_start = self.num_diffusion_steps
        graph_batch = self.make_pure_noise(graph_batch)
        device = graph_batch.xt.device
        sh = graph_batch.xt.shape
        B = sh[0]

        for t_int in range(t_start, self.sampling_step_size - 1, -self.sampling_step_size):
            t = torch.tensor([t_int] * B, device=device)
            graph_batch.t = t
            graph_batch.alpha_bar_t = self.alpha_bar(t, (1,))
            # model will generate pred_g, pred_v and pred_e. we only use pred_e here
            pred_x0 = model(graph_batch)[-1].squeeze(-1)
            t_new = t - self.sampling_step_size
            graph_batch.xt = self.reverse_diffusion(graph_batch.xt, t, t_new, self.eta_sampling, pred_x0=pred_x0, pred_noise=None, graph_batch=graph_batch)

        return graph_batch.xt