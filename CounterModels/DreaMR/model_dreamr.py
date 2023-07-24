from Utils.ema import EMA
from .Backbone.BolTDiffusion_adaIN.bolTDiffusion_adaIN import BolTDiffusion_adaIN
from .Backbone.BolTDiffusion_adaIN.hyperParams import getHyper_bolTDiffusion_adaIN
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import math

from collections import namedtuple
from einops import reduce
from tqdm import tqdm
from functools import partial

import copy

from .hyperParams import hyperDict_dreamr

# constants

ModelPrediction = namedtuple('ModelPrediction',
                             ['pred_noise', 'pred_x_start', 'pred_v'])

# helper functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1, ) * (len(x_shape) - 1)))


# for schedule specific helpers


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


# linear schedule, from ddgan
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


def get_sigma_schedule(T):
    n_timestep = T
    beta_min = 0.1
    beta_max = 20.0
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float32)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    return alpha_bars


def get_linear_logsnr(T):
    alphas = get_sigma_schedule(T)
    return log(alphas) - log(1 - alphas)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-(
        (t *
         (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas_cumprod = torch.clip(alphas_cumprod, 0, 1 - clamp_min)
    return alphas_cumprod.to(torch.float32)
    # betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    # return torch.clip(betas, 0, 0.999)


def get_sigma_logsnr(T):
    alphas = sigmoid_beta_schedule(T)
    print("alphas = {}".format(alphas))
    return log(alphas) - log(1 - alphas)


# cosine schedule
def phil_alpha_cosine_log_snr(T):

    timesteps = torch.tensor(list(range(T + 1)), dtype=torch.float32)
    print("from phil_alpha_cosine_log_snr timesteps.shape = {}".format(
        timesteps.shape))
    t = timesteps / T

    s = 0.008
    return -log(
        (torch.cos((t + s) / (1 + s) * math.pi * 0.5)**-2) - 1, eps=1e-5)


# for conversion between predictions


# OK
def predict_x_from_eps(z, eps, logsnr):
    """x = (z - sigma*eps)/alpha."""
    return torch.sqrt(1. + torch.exp(-logsnr)) * (
        z - eps / torch.sqrt(1. + torch.exp(logsnr)))


# OK
def predict_eps_from_x(z, x, logsnr):
    """eps = (z - alpha*x)/sigma."""
    return torch.sqrt(1. + torch.exp(logsnr)) * (
        z - x / torch.sqrt(1. + torch.exp(-logsnr)))


# OK
def predict_x_from_v(z, v, logsnr):
    alpha_t = torch.sqrt(torch.sigmoid(logsnr))
    sigma_t = torch.sqrt(torch.sigmoid(-logsnr))
    return alpha_t * z - sigma_t * v


# OK
def predict_v_from_x_and_eps(x, eps, logsnr):
    alpha_t = torch.sqrt(torch.sigmoid(logsnr))
    sigma_t = torch.sqrt(torch.sigmoid(-logsnr))
    return alpha_t * eps - sigma_t * x


class Model(nn.Module):

    def __init__(self, hyperParams_diffusion, details):

        super().__init__()

        backboneMethod = hyperParams_diffusion.backboneMethod
        self.backboneMethod = backboneMethod

        hyperParams_backbone = getHyper_bolTDiffusion_adaIN()
        Backbone = BolTDiffusion_adaIN

        self.hyperParams_backbone = hyperParams_backbone
        self.hyperParams_diffusion = hyperParams_diffusion

        self.details = details
        self.device = details.device

        self.noiseSchedule = hyperParams_diffusion.noiseSchedule

        self.backboneModel = Backbone(hyperParams_backbone,
                                      details).to(details.device)

        optimizer = torch.optim.Adam(self.backboneModel.parameters(),
                                     lr=hyperParams_backbone.lr)

        self.ema = EMA(optimizer, hyperParams_diffusion.emaDecay)
        print("ema decay = {}".format(hyperParams_diffusion.emaDecay))

        # DIFFUSION INITS

        self.distillationIndex = details.distillationIndex
        self.distillationStep = np.power(2, details.distillationIndex)

        intervalLength = hyperParams_diffusion.timesteps / details.expertCount

        self.timeInterval = [
            int(intervalLength * details.expertIndex),
            int(intervalLength * (details.expertIndex + 1))
        ]

        numberOfDistilledSteps = int(intervalLength / (self.distillationStep))

        self.legalTimeInstants = torch.linspace(int(self.timeInterval[0] - 1),
                                                int(self.timeInterval[1] - 1),
                                                steps=numberOfDistilledSteps +
                                                1).long()

        print("Distillation index = {}, legalTimeInstans = {}".format(
            details.distillationIndex, self.legalTimeInstants))

        self.timesteps = hyperParams_diffusion.timesteps
        self.samplingTimesteps = hyperParams_diffusion.samplingTimesteps

        self.objective = hyperParams_diffusion.objective
        self.loss_type = hyperParams_diffusion.lossType

        # initialize log snrs
        self.logsnr_linear = get_linear_logsnr(self.timesteps).to(self.device)
        self.logsnr_cosine_phil = phil_alpha_cosine_log_snr(self.timesteps).to(
            self.device)
        self.logsnr_sigmoid = get_sigma_logsnr(self.timesteps).to(self.device)

    def copy_from_teacher(self, expert_model):
        del self.backboneModel
        torch.cuda.empty_cache()

        self.backboneModel = copy.deepcopy(expert_model.backboneModel)

    def model_out(self, x, t):
        if ("bolT" in self.backboneMethod or "BolT" in self.backboneMethod):
            return self.backboneModel(x, t)[0]
        if ("unet1d" in self.backboneMethod):
            return self.backboneModel(x, t)

    def model_predictions_known(self, z, x_start, t, logsnr):

        maybe_clip = partial(torch.clamp, min=-6., max=6.)

        z = maybe_clip(z)
        x_start = maybe_clip(x_start)

        pred_noise = predict_eps_from_x(z, x_start, logsnr)
        v = predict_v_from_x_and_eps(x_start, pred_noise, logsnr)

        return pred_noise, x_start, v

    def model_predictions(self, z, t, logsnr, clip_x_start=False):

        model_output = self.model_out(z, t)
        # TODO CHECKOUT
        maybe_clip = partial(torch.clamp, min=-6.,
                             max=6.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = predict_x_from_eps(z, pred_noise, logsnr)
            x_start = maybe_clip(x_start)
            v = predict_v_from_x_and_eps(x_start, pred_noise, logsnr)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = predict_eps_from_x(z, x_start, logsnr)
            v = predict_v_from_x_and_eps(x_start, pred_noise, logsnr)

        elif self.objective == "pred_v":
            v = model_output
            x_start = predict_x_from_v(z, v, logsnr)
            pred_noise = predict_eps_from_x(z, x_start, logsnr)

        return ModelPrediction(pred_noise, x_start, v)

    @torch.no_grad()
    def ddim_sample_loop(self, inputRoiSignal):

        total_timesteps = self.timesteps
        samplingTimesteps = hyperDict_dreamr.samplingTimesteps

        if (self.distillationIndex > 0):
            times = self.legalTimeInstants

        else:
            factor = total_timesteps / samplingTimesteps

            remaining_timesteps = int(
                (self.timeInterval[1] - self.timeInterval[0]) / factor)

            times = torch.linspace(
                int(self.timeInterval[0] - 1),
                int(self.timeInterval[1] - 1),
                steps=remaining_timesteps + 1
            )  # [-1, 0, 1, 2, ..., T-1] when samplingTimesteps == total_timesteps

        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1],
                times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = inputRoiSignal

        for time, time_next in tqdm(time_pairs,
                                    desc='sampling loop time step',
                                    ascii=True):

            img, _ = self.ddim_sample(img, time, time_next)

        return img

    @torch.no_grad()
    def ddim_sample(self, img, time, time_next, grad=None, ddim_eta=0.0):
        batchSize = img.shape[0]
        device = img.device

        time_ = torch.full((batchSize, ),
                           time,
                           device=device,
                           dtype=torch.long)

        time_next_ = torch.full((batchSize, ),
                                time_next,
                                device=device,
                                dtype=torch.long)

        logsnr_time = self.get_log_snr(time_, img.shape)

        pred_noise, x_start, *_ = self.model_predictions(
            img, time_, logsnr_time)

        if time_next < 0:
            img = x_start
            return img, x_start

        logsnr_time_next = self.get_log_snr(time_next_, img.shape)

        #stdv_time = torch.sqrt(torch.sigmoid(-logsnr_time))
        alpha_time = torch.sqrt(torch.sigmoid(logsnr_time))

        stdv_time_next = torch.sqrt(torch.sigmoid(-logsnr_time_next))
        alpha_time_next = torch.sqrt(torch.sigmoid(logsnr_time_next))

        sigma = ddim_eta * ((1 - (alpha_time**2) / (alpha_time_next**2)) *
                            (1 - (alpha_time_next**2)) /
                            (1 - (alpha_time**2))).sqrt()

        c = (1 - alpha_time_next**2 - sigma**2).sqrt()  # stdv_time_next

        noise = torch.randn_like(img)

        img = x_start * alpha_time_next + \
            c * pred_noise + \
            sigma * noise

        if (grad != None):

            if (sigma[0] > 0.0):
                img += grad.detach() * sigma
            else:
                img += grad.detach()

        return img, x_start

    # @torch.no_grad()
    def ddim_denoise_from(self,
                          otherExperts,
                          roiSignals,
                          T_start,
                          isDime=False):

        # print("self.legalTimeInstants = {}, T_start = {}".format(
        #     self.legalTimeInstants, T_start))

        # if (T_start not in self.legalTimeInstants):
        #     raise "Error, this expert accepts t from one of the legal time instants : {}".format(
        #         self.legalTimeInstants)

        batchSize = roiSignals.shape[0]
        device = self.device
        total_timesteps = self.timesteps
        samplingTimesteps = hyperDict_dreamr.samplingTimesteps

        # total_timesteps = self.timesteps
        # samplingTimesteps = self.samplingTimesteps

        if (self.distillationIndex > 0):
            factor = self.distillationStep

            times = self.legalTimeInstants[:self.legalTimeInstants.cpu().
                                           tolist().index(T_start) + 1]

            # print("times = ", times)

        else:

            factor = total_timesteps / samplingTimesteps

            remaining_timesteps = int(
                (T_start - self.timeInterval[0]) / factor)

            times = torch.linspace(self.timeInterval[0] - 1,
                                   T_start,
                                   steps=remaining_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1],
                times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        if (len(time_pairs) == 0):

            time_pairs = [(T_start - 1, self.timeInterval[0] - 1)]
            # print("self.timeInterval = ", self.timeInterval)
            # print("T_start = ", T_start)
            # print("times = ", times)

        img = roiSignals

        print(
            "expertIndex = {}, time_pairs = ".format(self.details.expertIndex),
            time_pairs)

        for time, time_next in time_pairs:

            time_ = torch.full((batchSize, ),
                               time,
                               device=device,
                               dtype=torch.long)

            time_next_ = torch.full((batchSize, ),
                                    time_next,
                                    device=device,
                                    dtype=torch.long)

            logsnr_time = self.get_log_snr(time_, img.shape)

            pred_noise, x_start, *_ = self.model_predictions(
                img, time_, logsnr_time)

            if time_next < 0:
                img = x_start
                break

            logsnr_time_next = self.get_log_snr(time_next_, img.shape)

            #stdv_time = torch.sqrt(torch.sigmoid(-logsnr_time))
            alpha_time = torch.sqrt(torch.sigmoid(logsnr_time))

            stdv_time_next = torch.sqrt(torch.sigmoid(-logsnr_time_next))
            alpha_time_next = torch.sqrt(torch.sigmoid(logsnr_time_next))

            ddim_eta = hyperDict_dreamr.ddim_eta

            sigma = ddim_eta * ((1 - (alpha_time**2) / (alpha_time_next**2)) *
                                (1 - (alpha_time_next**2)) /
                                (1 - (alpha_time**2))).sqrt()

            c = (1 - alpha_time_next**2 - sigma**2).sqrt()  # stdv_time_next

            noise = torch.randn_like(img)

            img = x_start * alpha_time_next + \
                c * pred_noise + \
                sigma * noise

        if (time_next != -1):
            img = otherExperts[-1].ddim_denoise_from(otherExperts[:-1], img,
                                                     time_next)

        return img

    def genCounterfact(self,
                       otherExperts,
                       boldSignals,
                       guideModel,
                       targetClass,
                       T_start,
                       baseGuidanceScale=10.0):
        """
            otherExperts : [model_0, model_1, ..., model_selfIndex-1]
            method : ["tillEnd", "immediate"]
            boldSignals : (batchSize, T, R) -> this has to be torch tensor with correct device
        """

        # we already assume that the boldSignals are noisy
        bestMatchStartIndex = (self.legalTimeInstants - T_start).abs().argmin()

        if (self.distillationIndex != 0):
            times = self.legalTimeInstants[:bestMatchStartIndex + 1]

        else:
            factor = self.timesteps / self.samplingTimesteps

            remaining_timesteps = int(
                (T_start - self.timeInterval[0]) / factor)

            times = torch.linspace(self.timeInterval[0] - 1,
                                   T_start,
                                   steps=remaining_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1],
                times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = boldSignals.detach().clone()

        x_starts = [[], []]

        print("time_pairs = ", time_pairs)
        print("T_start = ", T_start)

        for time, time_next in tqdm(time_pairs, ncols=60, ascii=True):

            print("time = {}, time_next = {}".format(time, time_next))

            with torch.no_grad():

                x_start_good = self.ddim_denoise_from(
                    otherExperts, img, time, False)
                x_start = x_start_good

            with torch.no_grad():
                x_starts[0].append(x_start.detach().clone())
                x_starts[1].append((time, time_next))

            time_ = torch.full((x_start.shape[0], ),
                               time,
                               device=x_start.device,
                               dtype=torch.long)

            time_next_ = torch.full((x_start.shape[0], ),
                                    time_next,
                                    device=x_start.device,
                                    dtype=torch.long)

            x_start = x_start.detach()

            x_start.requires_grad_(True)

            logits = guideModel.getLogits(x_start.permute(0, 2, 1))
            log_probs = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)

            selected = log_probs[range(len(logits)), targetClass]

            grad_class = torch.autograd.grad(selected.sum(), x_start)[0]

            if (self.distillationIndex != 0):
                guidanceScale = (baseGuidanceScale) * (self.distillationStep /
                                                       np.power(2, 7))
            else:
                guidanceScale = baseGuidanceScale

            grad = grad_class * guidanceScale

            logsnr_time = self.get_log_snr(time_, img.shape)
            logsnr_time_next = self.get_log_snr(time_next_, img.shape)
            # logsnr_time_end = self.get_log_snr(time_end, img.shape)

            alpha_time = torch.sqrt(torch.sigmoid(logsnr_time))
            sigma_time = (1 - alpha_time**2)**0.5

            alpha_time_next = torch.sqrt(torch.sigmoid(logsnr_time_next))
            sigma_time_next = (1 - alpha_time_next**2)**0.5

            ddim_scale = (sigma_time**2 / alpha_time) * (
                -alpha_time * sigma_time_next / sigma_time +
                alpha_time_next)

            grad = grad * ddim_scale

            if (time_next < 0):
                img = x_start.detach()
                break

            with torch.no_grad():
                img_withgrad, _ = self.ddim_sample(img,
                                                   time,
                                                   time_next,
                                                   grad=grad.detach(),
                                                   ddim_eta=0.0)

                img = img_withgrad

            del grad
            del logits
            del x_start
            del log_probs
            del selected
            del probs

            torch.cuda.empty_cache()

        counterImage = img

        if (self.details.expertIndex == 0):
            probs = F.softmax(logits, dim=-1)
            return counterImage, x_starts, logits.detach().argmax(
                dim=1), probs
        else:
            return counterImage, x_starts

    @torch.no_grad()
    def sample(self, inputRoiSignal):
        return self.ddim_sample_loop(inputRoiSignal)

    def q_sample(self, x_start, logsnr, noise=None):

        device = x_start.device

        noise = default(
            noise,
            lambda: torch.randn_like(x_start).to(device),
        )

        return x_start * torch.sqrt(torch.sigmoid(logsnr)).to(
            device) + noise * torch.sqrt(torch.sigmoid(-logsnr)).to(device)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def get_log_snr(self, time, targetShape):
        """
            time : (batchSize, )
            targetX : (batchSize, ...)
        """

        if (self.noiseSchedule == "linear"):
            return extract(self.logsnr_linear, time + 1, targetShape)

        elif (self.noiseSchedule == "cosine"):
            return extract(self.logsnr_cosine_phil, time + 1, targetShape)

        elif (self.noiseSchedule == "sigmoid"):
            return extract(self.logsnr_sigmoid, time + 1, targetShape)

    def get_train_loss(self, x_start, t, noise=None, expertModel=None):

        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        logsnr = self.get_log_snr(t, x_start.shape)

        z = self.q_sample(x_start=x_start, logsnr=logsnr, noise=noise)

        # predict and take gradient step

        model_out = self.model_out(z, t)

        if (expertModel == None):
            if self.objective == 'pred_noise':
                target = noise
                x_start_predicted = predict_x_from_eps(z, model_out, logsnr)
            elif self.objective == 'pred_x0':
                target = x_start
                x_start_predicted = model_out
            elif self.objective == "pred_v":
                target = predict_v_from_x_and_eps(x_start, noise, logsnr)
                x_start_predicted = predict_x_from_v(z, model_out, logsnr)
            else:
                raise ValueError(f'unknown objective {self.objective}')
        else:
            x_target, eps_target, v_target = expertModel.getTarget(z, t)
            if self.objective == "pred_noise":
                target = eps_target
            elif self.objective == "pred_x0":
                target = x_target
            elif self.objective == "pred_v":
                target = v_target
            x_start_predicted = None

        loss = self.loss_fn(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        return loss.mean(), x_start_predicted

    def getTarget(self, z, t, clip_x_start=False):  # for the teacher model

        # print("Teacher with distillation index = {}, distillationStep = {}".
        #       format(self.distillationIndex, self.distillationStep))

        with torch.no_grad():

            # TODO CHECKOUT
            maybe_clip = partial(torch.clamp, min=-6.,
                                 max=6.) if clip_x_start else identity

            logsnr_t_start = self.get_log_snr(t, z.shape)
            model_out1 = self.model_out(z, t)

            if (self.objective == "pred_noise"):

                pred_noise = model_out1
                x_start = predict_x_from_eps(z, pred_noise, logsnr_t_start)
                x_start = maybe_clip(x_start)

            elif (self.objective == "pred_x0"):

                x_start = model_out1
                x_start = maybe_clip(x_start)
                pred_noise = predict_eps_from_x(z, x_start, logsnr_t_start)

            elif (self.objective == "pred_v"):

                v = model_out1
                x_start = predict_x_from_v(z, v, logsnr_t_start)
                pred_noise = predict_eps_from_x(z, x_start, logsnr_t_start)

            else:
                raise ValueError(f'unknown objective {self.objective}')

            logsnr_t_mid = self.get_log_snr(t - self.distillationStep, z.shape)
            a_mid = torch.sqrt(torch.sigmoid(logsnr_t_mid))
            std_mid = torch.sqrt(torch.sigmoid(-logsnr_t_mid))
            z_mid = a_mid * x_start + std_mid * pred_noise

            model_out2 = self.model_out(z_mid, t - self.distillationStep)

            if (self.objective == "pred_noise"):

                pred_noise = model_out2
                x_start = predict_x_from_eps(z, pred_noise, logsnr_t_start)
                x_start = maybe_clip(x_start)

            elif (self.objective == "pred_x0"):

                x_start = model_out2
                x_start = maybe_clip(x_start)
                pred_noise = predict_eps_from_x(z, x_start, logsnr_t_start)

            elif (self.objective == "pred_v"):

                v = model_out2
                x_start = predict_x_from_v(z, v, logsnr_t_start)
                pred_noise = predict_eps_from_x(z, x_start, logsnr_t_start)

            else:
                raise ValueError(f'unknown objective {self.objective}')

            logsnr_t_end = self.get_log_snr(t - 2 * self.distillationStep,
                                            z.shape)
            stdv_end = torch.sqrt(torch.sigmoid(-logsnr_t_end))
            a_end = torch.sqrt(torch.sigmoid(logsnr_t_end))
            z_end = a_end * x_start + stdv_end * pred_noise

            a_start = torch.sqrt(torch.sigmoid(logsnr_t_start))

            stdv_frac = torch.exp(
                0.5 * (F.softplus(logsnr_t_start) - F.softplus(logsnr_t_end)))

            x_target = (z_end - stdv_frac * z) / (a_end - stdv_frac * a_start)

            x_target = torch.where(
                t.reshape((t.shape[0], 1, 1)) == 0, x_start, x_target)

            eps_target = predict_eps_from_x(z, x_target, logsnr_t_start)

            v_target = predict_v_from_x_and_eps(x_target, eps_target,
                                                logsnr_t_start)

            return (x_target, eps_target, v_target)

    def step(self, roiSignals, device, model_teacher=None):
        """
        Input:
            roiSignals = (batchSize, R, T)
        """

        roiSignals = self.prepareInput(roiSignals, device)
        batchSize = roiSignals.shape[0]

        t = self.legalTimeInstants[torch.randint(0, len(
            self.legalTimeInstants), (batchSize, ))].to(device).long()

        # make sure there is not attempt to denoise starting image
        t[t == -1] = self.legalTimeInstants[1].long()

        loss, _ = self.get_train_loss(roiSignals, t, expertModel=model_teacher)

        self.ema.zero_grad()
        loss.backward()
        self.ema.step()

        loss = loss.detach().to("cpu")

        torch.cuda.empty_cache()

        return loss

    def swapWithEma(self):
        print("Swapping parameters with EMA")
        self.ema.swap_parameters_with_ema(False)

    def prepareInput(self, x, device):
        """
            x = (batchSize, N, T)
        """

        x = x.to(device).permute((0, 2, 1))

        return x
