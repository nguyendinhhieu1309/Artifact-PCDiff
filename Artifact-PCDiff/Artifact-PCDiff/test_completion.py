import argparse
import torch.nn as nn
import torch.utils.data

from torch.distributions import Normal
from .model.pvcnn_completion import ArtifactPVCNN2, PVCNN2Base
from .modules.artifact_method import (
    ReliabilityGatedFractureRefinement,
    interface_chamfer_loss,
    method_from_checkpoint,
)
from .utils.file_utils import *
from tqdm import tqdm
from .datasets.broken_data import BrokenDataset

'''
----- Models -----
'''

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 1]
    assert x.shape == means.shape == log_scales.shape
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)
    min_in = inv_stdv * (centered_x - .5)
    cdf_min = px0.cdf(min_in)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.ones_like(cdf_plus) * 1e-12))
    log_one_minus_cdf_min = torch.log(torch.max(1. - cdf_min, torch.ones_like(cdf_min) * 1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(x < 0.001, log_cdf_plus, torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(
        torch.max(cdf_delta, torch.ones_like(cdf_delta) * 1e-12))))
    assert log_probs.shape == x.shape
    return log_probs


class GaussianDiffusion:
    def __init__(self, betas, loss_type, model_mean_type, model_var_type, sv_points):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        #assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sv_points = sv_points
        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])

        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)

        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """ Diffuse the data (t == 0 means diffused for 1 step) """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)

        assert noise.shape == x_start.shape

        return (self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start + self._extract(
            self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """ Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0) """
        assert x_start.shape == x_t.shape
        posterior_mean = (self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                          self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t)
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t,
                                                       x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, denoise_fn, data, t, clip_denoised: bool, return_pred_xstart: bool):

        model_output = denoise_fn(data, t)[:, :, self.sv_points:]

        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = self.posterior_variance.to(data.device), \
                                                 self.posterior_log_variance_clipped.to(data.device)

            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(model_output)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(model_output)

        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data[:, :, self.sv_points:], t=t, eps=model_output)
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data[:, :, self.sv_points:], t=t)

        else:
            raise NotImplementedError(self.loss_type)

        assert model_mean.shape == x_recon.shape
        assert model_variance.shape == model_log_variance.shape

        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon

        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape

        return (self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps)

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t - pred_xstart) / \
               self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape)

    ''' 
    ----- DDPM sampling ----- 
    '''

    def p_sample(self, denoise_fn, data, t, noise_fn, clip_denoised=False, return_pred_xstart=False):
        """ Sample from the model """
        model_mean, _, model_log_variance = self.p_mean_variance(denoise_fn, data=data, t=t,
                                                                 clip_denoised=clip_denoised, return_pred_xstart=False)

        noise = noise_fn(size=model_mean.shape, dtype=model_mean.dtype, device=model_mean.device)

        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(model_mean.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        sample = torch.cat([data[:, :, :self.sv_points], sample], dim=-1)
        return sample

    def p_sample_loop(self, partial_x, denoise_fn, shape, device, noise_fn=torch.randn, clip_denoised=True,
                      keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps
        """

        assert isinstance(shape, (tuple, list))
        noise = noise_fn(size=shape, dtype=torch.float, device=device)

        img_t = torch.cat([partial_x, noise], dim=-1)

        iters = self.num_timesteps if not keep_running else len(self.betas)
        for t in tqdm(reversed(range(0, iters)), total=iters):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)

        assert img_t[:, :, self.sv_points:].shape == shape
        return img_t

    def p_sample_loop_trajectory(self, denoise_fn, shape, device, freq, noise_fn=torch.randn, clip_denoised=True,
                                 keep_running=False):
        """
        Generate samples, returning intermediate images
        Useful for visualizing how denoised images evolve over time
        Args:
          repeat_noise_steps (int): Number of denoising timesteps in which the same noise
            is used across the batch. If >= 0, the initial noise is the same for all batch elemements.
        """
        assert isinstance(shape, (tuple, list))

        total_steps = self.num_timesteps if not keep_running else len(self.betas)

        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        imgs = [img_t]
        for t in reversed(range(0, total_steps)):

            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)
            if t % freq == 0 or t == total_steps - 1:
                imgs.append(img_t)

        assert imgs[-1].shape == shape
        return imgs

    '''
    ----- DDIM sampling -----
    '''
    def ddim_sample(self, denoise_fn, data, t, noise_fn, clip_denoised=True, return_pred_xstart=True, eta=0.0):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """
        model_mean, _, _, x_start = self.p_mean_variance(denoise_fn, data=data, t=t, clip_denoised=clip_denoised,
                                                         return_pred_xstart=return_pred_xstart)

        eps = self._predict_eps_from_xstart(data[:, :, self.sv_points:], t, x_start)

        alpha_bar = self._extract(self.alphas_cumprod.to(data.device), t, data.shape)
        alpha_bar_prev = self._extract(self.alphas_cumprod_prev.to(data.device), t, data.shape)
        sigma = (eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev))

        noise = noise_fn(size=model_mean.shape, dtype=model_mean.dtype, device=model_mean.device)
        mean_pred = (x_start * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(data.shape) - 1))))  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        sample = torch.cat([data[:, :, :self.sv_points], sample], dim=-1)
        return sample

    def ddim_sample_loop(self, partial_x, denoise_fn, shape, device, noise_fn=torch.randn, clip_denoised=True,
                         sampling_steps=1000):

        assert isinstance(shape, (tuple, list))
        noise = noise_fn(size=shape, dtype=torch.float, device=device)

        img_t = torch.cat([partial_x, noise], dim=-1)

        ts = np.linspace(0, 999, sampling_steps).round().astype('int')
        ts = np.unique(ts)[::-1]

        for t in tqdm(ts):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.ddim_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                     clip_denoised=clip_denoised, return_pred_xstart=True)

        assert img_t[:, :, self.sv_points:].shape == shape
        return img_t

    '''
    ----- Losses -----
    '''

    def _vb_terms_bpd(self, denoise_fn, data_start, data_t, t, clip_denoised: bool, return_pred_xstart: bool):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=data_start[:, :, self.sv_points:], x_t=data_t[:, :, self.sv_points:], t=t)
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(denoise_fn, data=data_t, t=t,
                                                                              clip_denoised=clip_denoised,
                                                                              return_pred_xstart=True)

        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = kl.mean(dim=list(range(1, len(model_mean.shape)))) / np.log(2.)

        return (kl, pred_xstart) if return_pred_xstart else kl

    def p_losses(self, denoise_fn, data_start, t, noise=None):
        """ Training loss calculation """
        B, D, N = data_start.shape
        assert t.shape == torch.Size([B])

        if noise is None:
            noise = torch.randn(data_start[:, :, self.sv_points:].shape, dtype=data_start.dtype,
                                device=data_start.device)

        data_t = self.q_sample(x_start=data_start[:, :, self.sv_points:], t=t, noise=noise)

        if self.loss_type == 'mse':
            # predict the noise instead of x_start. seems to be weighted naturally like SNR
            eps_recon = denoise_fn(torch.cat([data_start[:, :, :self.sv_points], data_t], dim=-1), t)[:, :, self.sv_points:]
            losses = ((noise - eps_recon) ** 2).mean(dim=list(range(1, len(data_start.shape))))

        elif self.loss_type == 'kl':
            losses = self._vb_terms_bpd(denoise_fn=denoise_fn, data_start=data_start, data_t=data_t, t=t,
                                        clip_denoised=False, return_pred_xstart=False)

        else:
            raise NotImplementedError(self.loss_type)

        assert losses.shape == torch.Size([B])

        return losses

    '''
    ----- Debug -----
    '''

    def _prior_bpd(self, x_start):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps
            t_ = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(T - 1)
            qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=t_)
            kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=torch.tensor([0.]).to(qt_mean),
                                 logvar2=torch.tensor([0.]).to(qt_log_variance))
            assert kl_prior.shape == x_start.shape
            return kl_prior.mean(dim=list(range(1, len(kl_prior.shape)))) / np.log(2.)

    def calc_bpd_loop(self, denoise_fn, x_start, clip_denoised=True):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps

            vals_bt_, mse_bt_ = torch.zeros([B, T], device=x_start.device), torch.zeros([B, T], device=x_start.device)
            for t in reversed(range(T)):
                t_b = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(t)
                # Calculate VLB term at the current timestep
                data_t = torch.cat(
                    [x_start[:, :, :self.sv_points], self.q_sample(x_start=x_start[:, :, self.sv_points:], t=t_b)],
                    dim=-1)
                new_vals_b, pred_xstart = self._vb_terms_bpd(denoise_fn, data_start=x_start, data_t=data_t, t=t_b,
                                                             clip_denoised=clip_denoised, return_pred_xstart=True)

                # MSE for progressive prediction loss
                assert pred_xstart.shape == x_start[:, :, self.sv_points:].shape
                new_mse_b = ((pred_xstart - x_start[:, :, self.sv_points:]) ** 2).mean(
                    dim=list(range(1, len(pred_xstart.shape))))
                assert new_vals_b.shape == new_mse_b.shape == torch.Size([B])

                # Insert the calculated term into the tensor of all terms
                mask_bt = t_b[:, None] == torch.arange(T, device=t_b.device)[None, :].float()
                vals_bt_ = vals_bt_ * (~mask_bt) + new_vals_b[:, None] * mask_bt
                mse_bt_ = mse_bt_ * (~mask_bt) + new_mse_b[:, None] * mask_bt
                assert mask_bt.shape == vals_bt_.shape == vals_bt_.shape == torch.Size([B, T])

            prior_bpd_b = self._prior_bpd(x_start[:, :, self.sv_points:])
            total_bpd_b = vals_bt_.sum(dim=1) + prior_bpd_b
            assert vals_bt_.shape == mse_bt_.shape == torch.Size(
                [B, T]) and total_bpd_b.shape == prior_bpd_b.shape == torch.Size([B])
            return total_bpd_b.mean(), vals_bt_.mean(), prior_bpd_b.mean(), mse_bt_.mean()


class PVCNN2(PVCNN2Base):
    num_n = 128

    # Define set abstraction layers
    sa_blocks = [((32, 2, 32), (4096, 0.1, num_n, (32, 64))),
                 ((64, 3, 16), (2048, 0.2, num_n, (64, 128))),
                 ((128, 3, 8), (512, 0.4, num_n, (128, 256))),
                 (None, (160, 0.8, num_n, (256, 256, 512))),
                 ]

    # Define feature propagation layers
    fp_blocks = [((256, 256), (256, 3, 8)),
                 ((256, 256), (256, 3, 8)),
                 ((256, 128), (128, 2, 16)),
                 ((128, 128, 64), (64, 2, 32)),
                 ]

    def __init__(self, num_classes, sv_points, embed_dim, use_att, dropout, extra_feature_channels=3,
                 width_multiplier=1.0, voxel_resolution_multiplier=1.0):
        super().__init__(num_classes=num_classes, sv_points=sv_points, embed_dim=embed_dim, use_att=use_att,
                         dropout=dropout, extra_feature_channels=extra_feature_channels,
                         width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)


class Model(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type: str,
                 width_mult: float, vox_res_mult: float):
        super(Model, self).__init__()

        # Create diffusion (DDPM)
        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type,
                                           sv_points=(args.num_points-args.num_nn))
        self.sv_points = args.num_points - args.num_nn
        self.method = getattr(args, 'method', 'pcdiff')

        # Create diffusion (DDIM)
        if args.sampling_method == 'ddim':
            timesteps = np.linspace(0, args.time_num-1, args.sampling_steps).round().astype('int')
            timesteps = np.unique(timesteps)[::-1]
            timesteps = set(timesteps)

            last_alpha_cumprod = 1.0
            new_betas = []
            for i, alpha_cumprod in enumerate(self.diffusion.alphas_cumprod):
                if i in timesteps:
                    new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                    last_alpha_cumprod = alpha_cumprod
                else:
                    new_betas.append(0.0)
            new_betas = np.array(new_betas)
            self.diffusion = GaussianDiffusion(new_betas, loss_type, model_mean_type, model_var_type,
                                               sv_points=(args.num_points-args.num_nn))

        # Select the matching backbone from checkpoint metadata while leaving
        # the legacy PCDiff architecture untouched for existing checkpoints.
        backbone = ArtifactPVCNN2 if self.method == 'artifact_pcdiff' else PVCNN2
        self.model = backbone(num_classes=args.nc, sv_points=self.sv_points, embed_dim=args.embed_dim,
                              use_att=args.attention, dropout=args.dropout, extra_feature_channels=0,
                              width_multiplier=width_mult, voxel_resolution_multiplier=vox_res_mult)
        self.interface_points = int(getattr(args, 'interface_points', 64))
        self.interface_loss_weight = float(getattr(args, 'interface_loss_weight', 0.1))
        if self.method == 'artifact_pcdiff':
            self.fracture_refinement = ReliabilityGatedFractureRefinement(
                feature_dim=self.model.output_feature_channels,
                time_dim=args.embed_dim,
                attention_dim=int(getattr(args, 'attention_dim', 100)),
                fracture_points=int(getattr(args, 'fracture_points', 10)),
            )
        else:
            self.fracture_refinement = None

    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt = self.diffusion.calc_bpd_loop(self._denoise, x0, clip_denoised)

        return {'total_bpd_b': total_bpd_b,
                'terms_bpd': vals_bt,
                'prior_bpd_b': prior_bpd_b,
                'mse_bt': mse_bt}

    def _denoise(self, data, t):
        B, D, N = data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        if self.fracture_refinement is None:
            return self.model(data, t)

        base_noise, features, timestep_features = self.model(data, t, return_features=True)
        repair_noisy = data[:, :, self.sv_points:]
        repair_base_noise = base_noise[:, :, self.sv_points:]
        alpha_bar = self.diffusion._extract(
            self.diffusion.alphas_cumprod.to(data.device), t, repair_noisy.shape
        )
        repair_noise = self.fracture_refinement(
            fragment_xyz=data[:, :, :self.sv_points],
            noisy_repair=repair_noisy,
            base_noise=repair_base_noise,
            fragment_features=features[:, :, :self.sv_points],
            repair_features=features[:, :, self.sv_points:],
            timestep_features=timestep_features[:, :, self.sv_points:],
            alpha_bar=alpha_bar,
            timesteps=t,
            total_timesteps=self.diffusion.num_timesteps,
        )
        return torch.cat((base_noise[:, :, :self.sv_points], repair_noise), dim=-1)

    def get_loss_iter(self, data, noises=None):
        B, D, N = data.shape
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)

        if noises is not None:
            noises[t != 0] = torch.randn((t != 0).sum(), *noises.shape[1:]).to(noises)

        if self.fracture_refinement is None:
            losses = self.diffusion.p_losses(denoise_fn=self._denoise, data_start=data, t=t, noise=noises)
        else:
            fragment = data[:, :, :self.sv_points]
            target_repair = data[:, :, self.sv_points:]
            if noises is None:
                noises = torch.randn_like(target_repair)
            noisy_repair = self.diffusion.q_sample(x_start=target_repair, t=t, noise=noises)
            noisy_data = torch.cat((fragment, noisy_repair), dim=-1)
            predicted_noise = self._denoise(noisy_data, t)[:, :, self.sv_points:]
            diffusion_loss = ((noises - predicted_noise) ** 2).mean(dim=(1, 2))
            predicted_repair = self.diffusion._predict_xstart_from_eps(
                noisy_repair, t=t, eps=predicted_noise
            )
            interface_loss = interface_chamfer_loss(
                predicted_repair, target_repair, fragment, point_count=self.interface_points
            )
            losses = diffusion_loss + self.interface_loss_weight * interface_loss
        assert losses.shape == t.shape == torch.Size([B])
        return losses

    def gen_samples(self, partial_x, shape, device, noise_fn=torch.randn, clip_denoised=True, keep_running=False,
                    sampling_method='ddpm', sampling_steps=1000):
        if sampling_method == 'ddpm':
            return self.diffusion.p_sample_loop(partial_x, self._denoise, shape=shape, device=device, noise_fn=noise_fn,
                                                clip_denoised=clip_denoised, keep_running=keep_running)
        if sampling_method == 'ddim':
            return self.diffusion.ddim_sample_loop(partial_x, self._denoise, shape=shape, device=device,
                                                   noise_fn=noise_fn, clip_denoised=clip_denoised,
                                                   sampling_steps=sampling_steps)
        else:
            raise NotImplementedError("Not implemented. Use 'ddpm' or 'ddim'.")


    def train(self):
        self.model.train()
        if self.fracture_refinement is not None:
            self.fracture_refinement.train()

    def eval(self):
        self.model.eval()
        if self.fracture_refinement is not None:
            self.fracture_refinement.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)


def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)

    elif schedule_type == 'warm0.1':
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)

    elif schedule_type == 'warm0.2':
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)

    elif schedule_type == 'warm0.5':
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)

    else:
        raise NotImplementedError(schedule_type)

    return betas


#############################################################################
def get_dataset(path, num_points, num_nn, dataset):

    te_dataset = BrokenDataset(path=path, num_points=num_points, num_nn=num_nn, norm_mode='shape_bbox',
                                       eval=True)

    return te_dataset

def force_c_view(arr, dtype):
    if arr.flags['C_CONTIGUOUS']:
        return arr.view(dtype)
    if arr.flags['F_CONTIGUOUS']:
        arr = np.require(arr, requirements=['C'], dtype=np.float32)
        return arr.view(dtype)
    raise BufferError("To change the data type of an array it must be either C-CONTIGUOUS or F-CONTIGUOUS")


def make_difference(train, repair):
    train = train.numpy()
    in_type = train.dtype
    if in_type != repair.dtype:
        raise TypeError(f"Mismatching value types {in_type} and {repair.dtype}")

    dtype = [('batch', in_type),('point', in_type),('xyz', in_type)]

    in_arr = train.reshape(-1,3)
    sa_arr = repair.reshape(-1,3)
    reshaped_in = force_c_view(in_arr, dtype)
    reshaped_sa = force_c_view(sa_arr, dtype)

    reshaped_diff = np.setdiff1d(reshaped_sa, reshaped_in)

    diff_pc = reshaped_diff.view(in_type).reshape(-1,3)
    diff_pc = diff_pc[np.newaxis, :, :]
    return diff_pc


def evaluate_recon_mvr(opt, model, save_dir):
    test_dataset = get_dataset(opt.path, opt.num_points, opt.num_nn, opt.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.bs, shuffle=False,
                                                  num_workers=int(opt.workers), drop_last=False)

    model.eval()

    for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Reconstructing Samples'):
        pc = data['train_points'].transpose(1, 2).to("cuda:" + str(opt.gpu))
        batch_size = pc.shape[0]
        ensemble_num = opt.num_ens

        pc = pc.repeat(ensemble_num, 1, 1)
        noise_shape = torch.Size([batch_size * ensemble_num, 3, opt.num_nn])

        with torch.no_grad():
            sample = model.gen_samples(pc, noise_shape, pc.device, clip_denoised=False,
                                       sampling_method=opt.sampling_method, sampling_steps=opt.sampling_steps)
            sample = sample.detach().cpu()

        sample_np = sample.numpy().transpose(0, 2, 1)
        # repeat() lays out the batch in ensemble-sized blocks: [e0:b0..bN,
        # e1:b0..bN]. Regroup so each output directory receives its own samples.
        sample_np = sample_np.reshape(ensemble_num, batch_size, sample_np.shape[1], 3)
        sample_np = sample_np.transpose(1, 0, 2, 3)
        input_np = np.asarray(data['train_points'])
        shift_np = np.asarray(data['shift'])
        scale_np = np.asarray(data['scale'])

        for batch_idx in range(batch_size):
            # Dataset paths may use Windows backslashes even though this output
            # naming code historically split only on POSIX separators.
            path_parts = str(data['name'][batch_idx]).replace('\\', '/').rstrip('/').split('/')
            file_stem = os.path.splitext(path_parts[-1])[0]
            name = (path_parts[-2] if len(path_parts) > 1 else '') + file_stem
            sample_dir = os.path.join(save_dir, name)
            os.makedirs(sample_dir, exist_ok=True)

            object_samples = sample_np[batch_idx]
            object_input = input_np[batch_idx:batch_idx + 1]
            object_difference = make_difference(object_input, object_samples)
            np.save(os.path.join(sample_dir, 'input.npy'), object_input)
            np.save(os.path.join(sample_dir, 'sample.npy'), object_samples)
            np.save(os.path.join(sample_dir, 'difference.npy'), object_difference)
            np.save(os.path.join(sample_dir, 'shift.npy'), shift_np[batch_idx:batch_idx + 1])
            np.save(os.path.join(sample_dir, 'scale.npy'), scale_np[batch_idx:batch_idx + 1])
    return


def main(opt):
    output_dir = opt.eval_path
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    logger = setup_logging(output_dir)
    outf_syn, = setup_output_subdirs(output_dir, 'syn')

    # Resolve the architecture before constructing the model. Old checkpoints
    # have no method metadata and continue to load through the original path.
    resumed_param = torch.load(opt.model, map_location='cpu')
    if opt.method == 'auto':
        opt.method = method_from_checkpoint(resumed_param)
    artifact_config = resumed_param.get('artifact_config') or {}
    if opt.method == 'artifact_pcdiff':
        for key in ('fracture_points', 'attention_dim', 'interface_points', 'interface_loss_weight'):
            if key in artifact_config:
                setattr(opt, key, artifact_config[key])
    logger.info('Using model method: %s', opt.method)

    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type, width_mult=opt.width_mult,
                  vox_res_mult=opt.vox_res_mult)

    def _transform_(m):
        if isinstance(opt.gpu, int):
            return nn.parallel.DataParallel(m, device_ids=[opt.gpu])
        return nn.parallel.DataParallel(m)

    torch.cuda.set_device(opt.gpu)
    model = model.cuda(opt.gpu)

    if opt.distr_train:
        model.multi_gpu_wrapper(_transform_)

    model.eval()

    with torch.no_grad():
        logger.info("Perform sampling with:%s" % opt.model)

        model.load_state_dict(resumed_param['model_state'])

        if opt.eval_recon_mvr:
            # Evaluate generation
            evaluate_recon_mvr(opt, model, outf_syn)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help="set the path to the dataset here")
    parser.add_argument('--dataset', type=str, help="specify the used dataset ")
    parser.add_argument('--model', default='', required=True, help="path to model to sample from")
    parser.add_argument('--method', choices=['auto', 'pcdiff', 'artifact_pcdiff'], default='auto',
                        help="auto-detect checkpoint architecture; legacy PCDiff remains supported")
    parser.add_argument('--num_ens', type=int, default=1, help='number of samples for ensembling')
    parser.add_argument('--sampling_method', type=str, default='ddpm', help='ddpm or ddim')
    parser.add_argument('--sampling_steps', type=int, default=1000)

    parser.add_argument('--bs', type=int, default=1, help='input batch size')
    parser.add_argument('--workers', type=int, default=24, help='workers')
    parser.add_argument('--niter', type=int, default=100000, help='number of epochs to train for')  # not used

    parser.add_argument('--eval_recon_mvr', type=eval, default=True)
    parser.add_argument('--eval_saved', type=eval, default=True)

    parser.add_argument('--nc', type=int, default=3, help="dimension of one point (usually 3 for x, y,z)")
    parser.add_argument('--num_points', type=int, default=30720, help="number of points the point cloud should contain")
    parser.add_argument('--num_nn', type=int, default=3072, help="number of points that represent the implant")

    '''model'''
    parser.add_argument('--beta_start', type=float, default=0.0001)
    parser.add_argument('--beta_end', type=float, default=0.02)
    parser.add_argument('--schedule_type', type=str, default='linear')
    parser.add_argument('--time_num', type=int, default=1000)

    # params
    parser.add_argument('--attention', type=eval, default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', type=str, default='mse')
    parser.add_argument('--model_mean_type', type=str, default='eps')
    parser.add_argument('--model_var_type', type=str, default='fixedsmall')
    parser.add_argument('--vox_res_mult', type=float, default=1.0)
    parser.add_argument('--width_mult', type=float, default=1.0)
    parser.add_argument('--fracture_points', type=int, default=10)
    parser.add_argument('--attention_dim', type=int, default=100)
    parser.add_argument('--interface_points', type=int, default=64)
    parser.add_argument('--interface_loss_weight', type=float, default=0.1)
    parser.add_argument('--distr_train', type=eval, default=False)

    '''eval'''
    parser.add_argument('--eval_path', default='', required=True, help='set manual path to save the results')
    parser.add_argument('--manualSeed', default=48, type=int, help='random seed')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use. None means using all available GPUs.')

    opt = parser.parse_args()

    if torch.cuda.is_available():
        opt.cuda = True
    else:
        opt.cuda = False

    return opt


if __name__ == '__main__':
    opt = parse_args()

    main(opt)
