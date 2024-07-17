# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""

import torch
import numpy as np
from models.utils import from_flattened_numpy, to_flattened_numpy
from scipy import integrate
import logging
from typing import Any, Iterable, Tuple, Union

def init_sample(flow, sample_shape, z=None, device='cuda'):
    """Initialize samples."""
    if z is None:
        shape = sample_shape
        z0 = flow.get_z0(torch.zeros(shape, device=device), train=False).to(device)
        x = z0.detach().clone()
    else:
        shape = z.shape
        x = z
    return x, shape

@torch.no_grad()
def get_flow_sampler(flow, sample_shape, inverse_scaler, use_ode_sampler=None, clip_denoised=False, device='cuda'):
    """
    Get rectified flow sampler

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    if use_ode_sampler is None:
        use_ode_sampler = flow.use_ode_sampler

    @torch.no_grad()
    def one_step_sampler(model, z=None, reverse=False, **kwargs: Any,):
        """one_step_sampler.

        Args:
        model: A velocity model.
        z: If present, generate samples from latent code `z`.
        Returns:
        samples, number of function evaluations.
        """
        # Initial sample
        x, shape = init_sample(flow, sample_shape, z=z, device=device)
        
        ### one step
        eps = flow.eps # default: 1e-3
        t = torch.ones(shape[0], device=device) * eps
        pred = flow.model_forward_wrapper(model, x, t, **kwargs) ### Copy from models/utils.py 
        if 'x1' in flow.cfg.flow.consistency:   # predict x1
            x = pred
        else:
            x = x + pred * (flow.T - eps)
        x = inverse_scaler(x)   # [0, 1]
        if clip_denoised:
            x = torch.clamp(x, 0., 1.)
        nfe = 1
        return x, nfe
        
    @torch.no_grad()
    def euler_sampler(model, z=None, return_xh=False, reverse=False, progress=False, **kwargs: Any,):
        """The probability flow ODE sampler with simple Euler discretization.

        Args:
        model: A velocity model.
        z: If present, generate samples from latent code `z`.
        Returns:
        samples, number of function evaluations.
        """
        # Initial sample
        x, shape = init_sample(flow, sample_shape, z=z, device=device)
        
        ### Uniform
        dt = 1./flow.sample_N
        eps = flow.eps # default: 1e-3
        x_h = []
        t_h = []

        indices = range(flow.sample_N)
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        if reverse:
            # For reverse, we need to modify the time stepping
            for i in indices:
                x_h.append(x.cpu())
                num_t = flow.T - (i / flow.sample_N * (flow.T - eps))
                t = torch.ones(shape[0], device=device) * num_t
                t_h.append(t.cpu())
                pred = flow.model_forward_wrapper(model, x, t, **kwargs)
                x = x - pred * dt  # Note the negative sign here for reverse
        else:
            for i in indices:
                x_h.append(x.cpu())
                num_t = i / flow.sample_N * (flow.T - eps) + eps
                t = torch.ones(shape[0], device=device) * num_t
                t_h.append(t.cpu())
                pred = flow.model_forward_wrapper(model, x, t, **kwargs)
                x = x + pred * dt
        x_h.append(x.cpu())
        t_h.append(t.cpu())
        t_h = [t_[:1] for t_ in t_h]
        t_h = torch.cat(t_h, dim=0)
        t_h = np.array(t_h)
        x = inverse_scaler(x)   # [0, 1]
        if clip_denoised:
            x = torch.clamp(x, 0., 1.)
        nfe = flow.sample_N
        if return_xh:
            # to be consistent with rk45_sampler
            return x, nfe, (x_h, t_h)
        else:
            return x, nfe
        
    @torch.no_grad()
    def heun_sampler(model, z=None, return_xh=False, reverse=False, progress=False, rho=1., **kwargs: Any,):
        """The probability flow ODE sampler with simple Heun discretization.

        Args:
        model: A velocity model.
        z: If present, generate samples from latent code `z`.
        Returns:
        samples, number of function evaluations.
        """
        assert not reverse, 'Not Implemented!'
        # Initial sample
        x, shape = init_sample(flow, sample_shape, z=z, device=device)
        
        ### Uniform
        # dt = 1./flow.sample_N
        eps = flow.eps # default: 1e-3
        x_h = []
        
        indices = torch.arange(flow.sample_N+1, device=device)
        timesteps = eps**rho + indices / max(flow.sample_N, 1) * (
            flow.T**rho - eps**rho
        )
        timesteps = timesteps**(1/rho)
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        for i in indices:
            x_h.append(x.cpu())
            num_t = timesteps[i]
            num_t_next = timesteps[i+1] if i+1 < flow.sample_N else flow.T
            dt = num_t_next - num_t
            t = torch.ones(shape[0], device=device) * num_t
            pred = flow.model_forward_wrapper(model, x, t, **kwargs) ### Copy from models/utils.py 

            if num_t_next == flow.T:
                # Euler method
                x = x + pred * dt
            else:
                # Heun's method
                x_2 = x + pred * dt
                t_2 = torch.ones(shape[0], device=device) * num_t_next
                # x_2 = x + d * dt
                pred_2 = flow.model_forward_wrapper(model, x_2, t_2, **kwargs)
                pred_prime = (pred + pred_2) / 2
                x = x + pred_prime * dt
        x_h.append(x.cpu())
        x = inverse_scaler(x)
        if clip_denoised:
            x = torch.clamp(x, 0., 1.)
        nfe = flow.sample_N * 2 - 1
        if return_xh:
            return x, nfe, x_h
        else:
            return x, nfe
        
    @torch.no_grad()
    def rk45_sampler(model, z=None, return_xh=False, reverse=False, **kwargs: Any,):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
        model: A velocity model.
        z: If present, generate samples from latent code `z`.
        Returns:
        samples, number of function evaluations.
        """
        rtol = atol = flow.ode_tol
        method = 'RK45'
        eps = flow.eps # default: 1e-3

        # Initial sample
        x, shape = init_sample(flow, sample_shape, z=z, device=device)

        @torch.no_grad()
        def ode_func(t, x):
            x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
            vec_t = torch.ones(shape[0], device=x.device) * t
            drift = flow.model_forward_wrapper(model, x, vec_t, **kwargs)
            # drift = model(x, vec_t*999)
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        if reverse:
            solution = integrate.solve_ivp(ode_func, (flow.T, eps), to_flattened_numpy(x),
                                            rtol=rtol, atol=atol, method=method)
        else:
            solution = integrate.solve_ivp(ode_func, (eps, flow.T), to_flattened_numpy(x),
                                            rtol=rtol, atol=atol, method=method)
        nfe = solution.nfev
        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
        x_h = torch.tensor(solution.y.T)
        x_h = [xi.reshape(shape).type(torch.float32) for xi in x_h]
        x = inverse_scaler(x)   # [0, 1]
        if clip_denoised:
            x = torch.clamp(x, 0., 1.)
        if return_xh:
            return x, nfe, (x_h, solution.t)
        else:
            return x, nfe
    
    @torch.no_grad()
    def stochastic_iterative_sampler(model, z=None, reverse=False, **kwargs: Any,):
        """The stochastic_iterative sampler proposed in consistency models.

        Args:
        model: A velocity model.
        z: If present, generate samples from latent code `z`.
        Returns:
        samples, number of function evaluations.
        """
        ### reference: https://github.com/openai/consistency_models/blob/main/cm/karras_diffusion.py#L657
        assert not reverse, 'Not Implemented!'
        # Initial sample
        x, shape = init_sample(flow, sample_shape, z=z, device=device)

        ### Uniform
        dt = 1./flow.sample_N
        eps = flow.eps # default: 1e-3
        
        for i in range(flow.sample_N):
            num_t = i / flow.sample_N * (flow.T - eps) + eps
            num_t_next = (i+1) / flow.sample_N * (flow.T - eps) + eps
            t = torch.ones(shape[0], device=device) * num_t
            pred = flow.model_forward_wrapper(model, x, t, **kwargs) ### Copy from models/utils.py 
            if 'x1' in flow.cfg.flow.consistency:
                x1 = pred
            else:
                x1 = x + pred * (1-num_t)
            x = num_t_next * x1 + (1 - num_t_next) * torch.randn_like(x1)

        x = inverse_scaler(x)   # [0, 1]
        if clip_denoised:
            x = torch.clamp(x, 0., 1.)

        return x, flow.sample_N
        

    if use_ode_sampler == 'one_step':
        sample_N = 1
    elif use_ode_sampler == 'rk45':
        sample_N = "adaptive"
    else:
        sample_N = flow.sample_N
    logging.info(f'Type of Sampler: {use_ode_sampler}; sample_N: {sample_N}')
    if use_ode_sampler == 'one_step':
        return one_step_sampler
    elif use_ode_sampler == 'euler':
        return euler_sampler
    elif use_ode_sampler == 'heun':
        return heun_sampler
    elif use_ode_sampler == 'rk45':
        return rk45_sampler
    elif use_ode_sampler == 'cm_stochastic':
        return stochastic_iterative_sampler
    else:
        assert False, 'Not Implemented!'
