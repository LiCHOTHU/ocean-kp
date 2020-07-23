import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import rlkit.torch.pytorch_util as ptu
from pytorch3d.transforms import quaternion_invert, quaternion_apply

import pdb
import os

eps = 1e-11


def _bad_product_of_cateogircal(z_means):  # it cause divide by zero because numerical issues.
    # print (z_means)
    z_mean = torch.prod(z_means, dim=0)
    # print (z_mean)
    return z_mean / torch.sum(z_mean)


def _product_of_cateogircal(z_means):
    z_means = torch.log(z_means + eps)
    z_mean = torch.sum(z_means, dim=0)
    cc = torch.max(z_mean).detach()
    z_mean -= cc
    z_mean = torch.exp(z_mean)
    return z_mean / torch.sum(z_mean)


def _product_of_cateogircal_all(z_means):
    z_means = torch.log(z_means + eps)
    z_mean = torch.sum(z_means, dim=-2)
    cc = torch.max(z_mean).detach()
    z_mean -= cc
    z_mean = torch.exp(z_mean)
    return F.normalize(z_mean, p=1, dim=-1)


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _product_of_gaussians_all(mus, sigmas_squared):
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=-2)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=-2)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2


def read_dim(s):
    a, b, c, d, e = s.split('.')
    return [int(a), int(b), int(c), int(d), int(e)]


class PEARLAgent(nn.Module):

    def __init__(self,
                 # latent_dim,
                 # num_cat,
                 # cont_latent_dim,
                 # dir_latent_dim,
                 # num_dir,
                 KeyNet,
                 GNN,
                 context_encoder,
                 context_encoder2,
                 recurrent_context_encoder,
                 n_train_tasks,
                 global_latent,
                 vrnn_latent,
                 policy,
                 temperature,
                 unitkl,
                 alpha,
                 # prior,
                 constraint,
                 vrnn_constraint,
                 var,
                 r_alpha,
                 r_var,
                 rnn,
                 temp_res,
                 rnn_sample,
                 **kwargs
                 ):
        super().__init__()
        self.cont_latent_dim, self.num_cat, self.latent_dim, self.num_dir, self.dir_latent_dim = read_dim(global_latent)
        if recurrent_context_encoder != None:
            self.r_cont_dim, self.r_n_cat, self.r_cat_dim, self.r_n_dir, self.r_dir_dim = read_dim(vrnn_latent)
        # print (latent_dim, num_cat, cont_latent_dim, dir_latent_dim, self.num_dir)

        self.n_train_tasks = n_train_tasks
        self.KeyNet = KeyNet
        self.GNN = GNN
        self.kp_list = [None for _ in range(self.n_train_tasks)]
        self.tasks_z_list = [None for _ in range(self.n_train_tasks)]
        # self.kp_criterion = kp_criterion

        self.context_encoder = context_encoder
        self.context_encoder2 = context_encoder2
        self.recurrent_context_encoder = recurrent_context_encoder
        self.policy = policy
        self.temperature = temperature
        self.unitkl = unitkl

        # self.prior = prior
        self.alpha = alpha
        self.constraint = constraint
        self.vrnn_constraint = vrnn_constraint
        self.var = var
        self.r_alpha = r_alpha
        self.r_var = r_var
        self.rnn = rnn

        self.temp_res = temp_res
        self.rnn_sample = rnn_sample
        self.n_global, self.n_local, self.n_infer = 0, 0, 0

        self.recurrent = kwargs['recurrent']
        self.glob = kwargs['glob']
        self.use_ib = kwargs['use_information_bottleneck']
        self.sparse_rewards = kwargs['sparse_rewards']

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        if self.glob:
            self.register_buffer('z', torch.zeros(1,
                                                  self.cont_latent_dim + self.latent_dim * self.num_cat + self.dir_latent_dim * self.num_dir))
            if self.latent_dim > 0:
                self.register_buffer('z_means', torch.zeros(1, self.latent_dim))
            if self.cont_latent_dim > 0:
                self.register_buffer('z_c_means', torch.zeros(1, self.cont_latent_dim))
                self.register_buffer('z_c_vars', torch.ones(1, self.cont_latent_dim))
            if self.dir_latent_dim > 0:
                self.register_buffer('z_d_means', torch.zeros(1, self.dir_latent_dim))
                self.register_buffer('z_d_vars', torch.ones(1, self.dir_latent_dim))

        if self.recurrent:
            self.register_buffer('seq_z', torch.zeros(1,
                                                      self.r_cont_dim + self.r_cat_dim * self.r_n_cat + self.r_dir_dim * self.r_n_dir))
            # self.register_buffer('hn', torch.zeros(1, self.recurrent_context_encoder.hidden_dim))
            # self.register_buffer('cn', torch.zeros(1, self.recurrent_context_encoder.hidden_dim))
            z_cat_prior, z_cont_prior, z_dir_prior = ptu.FloatTensor(), ptu.FloatTensor(), ptu.FloatTensor()
            if self.r_cat_dim > 0:
                self.register_buffer('seq_z_cat', torch.zeros(1, self.r_cat_dim))
                self.seq_z_next_cat = None
                z_cat_prior = ptu.ones(self.r_cat_dim * self.r_n_cat) / self.r_cat_dim
            if self.r_dir_dim > 0:
                if self.vrnn_constraint == 'logitnormal':
                    self.register_buffer('seq_z_dir_mean', torch.zeros(1, self.r_dir_dim))
                    self.register_buffer('seq_z_dir_var', torch.ones(1, self.r_dir_dim))
                    self.seq_z_next_dir_mean = None
                    self.seq_z_next_dir_var = None
                    z_dir_prior_mean = ptu.zeros(self.r_n_dir * self.r_dir_dim)
                    z_dir_prior_var = ptu.ones(self.r_n_dir * self.r_dir_dim) * self.r_var
                    z_dir_prior = torch.cat([z_dir_prior_mean, z_dir_prior_var])
                elif self.vrnn_constraint == 'dirichlet':
                    self.register_buffer('seq_z_dir', torch.zeros(1, self.r_dir_dim))
                    self.seq_z_next_dir = None
                    z_dir_prior = ptu.ones(self.r_n_dir * self.r_dir_dim) * self.r_alpha
            if self.r_cont_dim > 0:
                self.register_buffer('seq_z_cont_mean', torch.zeros(1, self.r_cont_dim))
                self.register_buffer('seq_z_cont_var', torch.zeros(1, self.r_cont_dim))
                self.seq_z_next_cont_mean = None
                self.seq_z_next_cont_var = None
                z_cont_prior = torch.cat([ptu.zeros(self.r_cont_dim), ptu.ones(self.r_cont_dim)])
            self.seq_z_prior = torch.cat([z_cat_prior, z_cont_prior, z_dir_prior])

        self.clear_z()

        # a = self.compute_kl_div()
        # self.infer_posterior(None)

    def clear_z(self, num_tasks=1, batch_size=1,
                traj_batch_size=1):  # ! check when it happens, because it looks fishy for batch size part
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        if self.glob:
            # reset distribution over z to the prior
            if self.latent_dim > 0:
                self.z_means = ptu.ones(num_tasks * self.num_cat, self.latent_dim) / self.latent_dim
            if self.cont_latent_dim > 0:
                self.z_c_means = ptu.zeros(num_tasks, self.cont_latent_dim)
                self.z_c_vars = ptu.ones(num_tasks, self.cont_latent_dim)
            if self.dir_latent_dim > 0:
                if self.constraint == 'logitnormal':
                    self.z_d_means = ptu.zeros(num_tasks * self.num_dir, self.dir_latent_dim)
                    self.z_d_vars = ptu.ones(num_tasks * self.num_dir, self.dir_latent_dim) * self.var
                else:
                    self.z_d_means = ptu.ones(num_tasks * self.num_dir, self.dir_latent_dim) * self.alpha

                    # sample a new z from the prior
            self.sample_z()

        if self.recurrent:
            if self.r_cat_dim > 0:
                self.seq_z_cat = ptu.ones(num_tasks * batch_size * self.r_n_cat, self.r_cat_dim) / self.r_cat_dim
                self.seq_z_next_cat = None
            if self.r_cont_dim > 0:
                self.seq_z_cont_mean = ptu.zeros(num_tasks * batch_size, self.r_cont_dim)
                self.seq_z_cont_var = ptu.ones(num_tasks * batch_size, self.r_cont_dim)
                self.seq_z_next_cont_mean = None
                self.seq_z_next_cont_var = None
            if self.r_dir_dim > 0:
                if self.vrnn_constraint == 'logitnormal':
                    self.seq_z_dir_mean = ptu.zeros(num_tasks * batch_size * self.r_n_dir, self.r_dir_dim)
                    self.seq_z_dir_var = ptu.ones(num_tasks * batch_size * self.r_n_dir, self.r_dir_dim) * self.r_var
                    self.seq_z_next_dir_mean = None
                    self.seq_z_next_dir_var = None
                elif self.vrnn_constraint == 'dirichlet':
                    self.seq_z_dir = ptu.ones(num_tasks * batch_size * self.r_n_dir, self.r_dir_dim) * self.r_alpha
                    self.seq_z_next_dir = None

            self.sample_sequence_z()

        # reset the context collected so far
        self.context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        if self.context_encoder != None:
            self.context_encoder.reset(num_tasks)
        if self.recurrent_context_encoder != None:
            self.recurrent_context_encoder.reset(num_tasks * traj_batch_size)

    def detach_z(self):
        ''' disable backprop through z '''
        if self.glob:
            self.z = self.z.detach()
        if self.recurrent:
            self.recurrent_context_encoder.hn = self.recurrent_context_encoder.hn.detach()
            self.recurrent_context_encoder.cn = self.recurrent_context_encoder.cn.detach()
            self.seq_z = self.seq_z.detach()

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        if False:
            prior = torch.distributions.Categorical(ptu.ones(self.latent_dim) / self.latent_dim)
            if self.unitkl:
                posteriors = [torch.distributions.Categorical(mu) for mu in torch.unbind(self.z_means_all)]
            else:
                posteriors = [torch.distributions.Categorical(mu) for mu in torch.unbind(self.z_means)]
            # prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
            # posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
            kl_div_sum = torch.sum(torch.stack(kl_divs))
        if True:
            kl_div_cont, kl_div_disc, kl_div_dir = ptu.FloatTensor([0.]).mean(), ptu.FloatTensor(
                [0.]).mean(), ptu.FloatTensor([0.]).mean()
            kl_div_seq_cont, kl_div_seq_disc, kl_div_seq_dir = ptu.FloatTensor([0.]).mean(), ptu.FloatTensor(
                [0.]).mean(), ptu.FloatTensor([0.]).mean()
            if self.unitkl:
                assert False
                if self.latent_dim > 0:
                    kl_div_disc = torch.sum(self.z_means_all * torch.log((self.z_means_all + eps) * self.latent_dim))
                if self.dir_latent_dim > 0 and self.constraint == 'logitnormal' and False:  # actually gaussians should not have kl divergence in each step!!
                    prior = torch.distributions.Normal(ptu.zeros(self.dir_latent_dim),
                                                       ptu.ones(self.dir_latent_dim) * np.sqrt(self.var))
                    posteriors = torch.distributions.Normal(self.z_d_means_all, torch.sqrt(self.z_d_vars_all))
                    kl_div_dir = torch.sum(torch.distributions.kl.kl_divergence(posteriors, prior))
                    # prior = torch.distributions.Dirichlet(ptu.ones(self.latent_dim))
                    # posteriors = [torch.distributions.Dirichlet()]
                if self.cont_latent_dim > 0 and False:
                    kl_div_cont = torch.sum(0.5 * (-torch.log(
                        self.z_c_vars_all) + self.z_c_vars_all + self.z_c_means_all * self.z_c_means_all - 1))
            else:
                if self.glob:
                    if self.latent_dim > 0:
                        kl_div_disc = torch.sum(self.z_means * torch.log((self.z_means + eps) * self.latent_dim))
                    if self.dir_latent_dim > 0:
                        if self.constraint == 'deepsets':
                            prior = torch.distributions.Dirichlet(ptu.ones(self.latent_dim) * self.alpha)
                            posteriors = torch.distributions.Dirichlet(self.z_d_means)
                            kl_div_dir = torch.sum(torch.distributions.kl.kl_divergence(posteriors, prior))
                        elif self.constraint == 'logitnormal':
                            prior = torch.distributions.Normal(ptu.zeros(self.dir_latent_dim),
                                                               ptu.ones(self.dir_latent_dim) * np.sqrt(self.var))
                            posteriors = torch.distributions.Normal(self.z_d_means, torch.sqrt(self.z_d_vars))
                            kl_div_dir = torch.sum(torch.distributions.kl.kl_divergence(posteriors, prior))
                    if self.cont_latent_dim > 0:
                        kl_div_cont = torch.sum(0.5 * (-torch.log(
                            self.z_c_vars) + self.z_c_vars + self.z_c_means * self.z_c_means - 1))  # ! still needs verify

                if self.recurrent:
                    if self.rnn == 'rnn':
                        if self.r_cat_dim > 0:
                            assert type(self.seq_z_next_cat) != type(None)
                            kl_div_seq_disc = torch.sum(
                                self.seq_z_cat * torch.log((self.seq_z_cat + eps) * self.r_cat_dim)) \
                                              + torch.sum(
                                self.seq_z_next_cat * torch.log((self.seq_z_next_cat + eps) * self.r_cat_dim))
                        if self.r_dir_dim > 0:
                            if self.vrnn_constraint == 'dirichlet':
                                assert type(self.seq_z_next_dir) != type(None)
                                prior = torch.distributions.Dirichlet(ptu.ones(self.r_dir_dim) * self.r_alpha)
                                posteriors = torch.distributions.Dirichlet(self.seq_z_dir)
                                posteriors_next = torch.distributions.Dirichlet(self.seq_z_next_dir)
                                kl_div_seq_dir = torch.sum(torch.distributions.kl.kl_divergence(posteriors, prior)) \
                                                 + torch.sum(
                                    torch.distributions.kl.kl_divergence(posteriors_next, prior))
                            elif self.vrnn_constraint == 'logitnormal':
                                assert type(self.seq_z_next_dir_mean) != type(None)
                                prior = torch.distributions.Normal(ptu.zeros(self.r_dir_dim),
                                                                   ptu.ones(self.r_dir_dim) * np.sqrt(self.r_var))
                                posteriors = torch.distributions.Normal(self.seq_z_dir_mean,
                                                                        torch.sqrt(self.seq_z_dir_var))
                                posteriors_next = torch.distributions.Normal(self.seq_z_next_dir_mean,
                                                                             torch.sqrt(self.seq_z_next_dir_var))
                                kl_div_seq_dir = torch.sum(torch.distributions.kl.kl_divergence(posteriors, prior)) \
                                                 + torch.sum(
                                    torch.distributions.kl.kl_divergence(posteriors_next, prior))
                        if self.r_cont_dim > 0:
                            kl_div_seq_cont = torch.sum(0.5 * (-torch.log(
                                self.seq_z_cont_var) + self.seq_z_cont_var + self.seq_z_cont_mean * self.seq_z_cont_mean - 1)) \
                                              + torch.sum(0.5 * (-torch.log(
                                self.seq_z_next_cont_var) + self.seq_z_next_cont_var + self.seq_z_next_cont_mean * self.seq_z_next_cont_mean - 1))
                    elif self.rnn == 'vrnn':
                        kl_div_seq_disc, kl_div_seq_cont, kl_div_seq_dir = self.recurrent_context_encoder.compute_kl_div()

        # return kl_div_disc, kl_div_cont
        return kl_div_disc, kl_div_cont, kl_div_dir, kl_div_seq_disc, kl_div_seq_cont, kl_div_seq_dir

    def infer_posterior(self, context, ff=False):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        # if ff:
        #     pdb.set_trace()
        # print (context.size())
        # print (context)
        if self.dir_latent_dim > 0 and self.constraint == 'deepsets':
            # context
            params = self.context_encoder2(context)
            print(params.size())
            exit(-1)
            if self.latent_dim + self.cont_latent_dim > 0:
                params = self.context_encoder(context)
                params = params.view(context.size(0), -1, self.num_cat * self.latent_dim + 2 * self.cont_latent_dim)
                print(params.size()[1])
        else:
            params = self.context_encoder(context)
            params = params.view(context.size(0), -1,
                                 self.num_cat * self.latent_dim + 2 * self.cont_latent_dim + self.num_dir * self.dir_latent_dim * 2)

        if self.latent_dim > 0:
            params_disc = params[..., :self.num_cat * self.latent_dim]
            params_disc = params_disc.view(context.size(0), -1, self.num_cat, self.latent_dim)
            params_disc = params_disc.transpose(1, 2)
            mu = F.softmax(params_disc, dim=-1)
            if self.unitkl:
                self.z_means_all = mu.view(-1, self.latent_dim)
            # print (mu)
            # print (params_disc.size())
            if False:
                z_params = []
                for i in torch.unbind(mu):
                    for j in torch.unbind(i):
                        z_params.append(_product_of_cateogircal(j))
                # print (z_params)
                self.z_means = torch.stack(z_params)
            if True:
                self.z_means = _product_of_cateogircal_all(mu).view(-1, self.latent_dim)

        if self.cont_latent_dim > 0:
            params_cont = params[...,
                          self.num_cat * self.latent_dim:self.num_cat * self.latent_dim + 2 * self.cont_latent_dim]
            mu_c = params_cont[..., :self.cont_latent_dim]
            sigma_squared_c = F.softplus(params_cont[..., self.cont_latent_dim:])
            if self.unitkl:
                self.z_c_means_all = mu_c.view(-1, self.cont_latent_dim)
                self.z_c_vars_all = sigma_squared_c.view(-1, self.cont_latent_dim)
            if False:
                z_params_c = [_product_of_gaussians(m, s) for m, s in
                              zip(torch.unbind(mu_c), torch.unbind(sigma_squared_c))]
                self.z_c_means = torch.stack([p[0] for p in z_params_c])
                self.z_c_vars = torch.stack([p[1] for p in z_params_c])
            if True:
                self.z_c_means, self.z_c_vars = _product_of_gaussians_all(mu_c, sigma_squared_c)

        if self.dir_latent_dim > 0 and self.constraint == 'logitnormal':
            params_dir = params[..., self.num_cat * self.latent_dim + 2 * self.cont_latent_dim:]
            params_dir = params_dir.view(context.size(0), -1, self.num_dir, self.dir_latent_dim * 2)
            params_dir = params_dir.transpose(1, 2)
            mu_d = params_dir[..., :self.dir_latent_dim]
            sigma_squared_d = F.softplus(params_dir[..., self.dir_latent_dim:])
            if self.unitkl:
                self.z_d_means_all = mu_d.view(-1, self.dir_latent_dim)
                self.z_d_vars_all = sigma_squared_d.view(-1, self.dir_latent_dim)
            self.z_d_means, self.z_d_vars = _product_of_gaussians_all(mu_d, sigma_squared_d)
            self.z_d_means = self.z_d_means.view(-1, self.dir_latent_dim)
            self.z_d_vars = self.z_d_vars.view(-1, self.dir_latent_dim)

        self.sample_z()

    def sample_z(self):
        if True:  # !! I think this one might be quicker, test it before push
            z, z_c, z_d = ptu.FloatTensor(), ptu.FloatTensor(), ptu.FloatTensor()
            if self.latent_dim > 0:
                gumbel = torch.distributions.Gumbel(ptu.FloatTensor([0]), ptu.FloatTensor([1.0])).sample(
                    self.z_means.size()).squeeze(-1)
                log_z = torch.log(self.z_means + eps)
                logit = (log_z + gumbel) / self.temperature
                z = F.softmax(logit, dim=1).view(-1, self.num_cat, self.latent_dim).view(-1,
                                                                                         self.num_cat * self.latent_dim)
            if self.cont_latent_dim > 0:
                normal = torch.distributions.Normal(ptu.FloatTensor([0.]), ptu.FloatTensor([1.])).sample(
                    self.z_c_means.size()).squeeze(-1)
                z_c = self.z_c_means + torch.sqrt(self.z_c_vars) * normal
            if self.dir_latent_dim > 0:
                if self.constraint == 'deepsets':
                    z_d = torch.distributions.Dirichlet(self.z_d_means).rsample() \
                        .view(-1, self.num_dir, self.dir_latent_dim).view(-1, self.num_dir * self.dir_latent_dim)
                elif self.constraint == 'logitnormal':
                    normal = torch.distributions.Normal(ptu.FloatTensor([0.]), ptu.FloatTensor([1.])).sample(
                        self.z_d_means.size()).squeeze(-1)
                    z_d = F.softmax(self.z_d_means + torch.sqrt(self.z_d_vars) * normal, dim=-1) \
                        .view(-1, self.num_dir, self.dir_latent_dim).view(-1, self.num_dir * self.dir_latent_dim)

            self.z = torch.cat([z, z_c, z_d], dim=-1)

        else:  # slow implementation deprecated
            soft_cats = [torch.distributions.RelaxedOneHotCategorical(ptu.FloatTensor([self.temperature]), m) for m in
                         torch.unbind(self.z_means)]
            z = [d.rsample() for d in soft_cats]
            self.z = torch.stack(z).view(-1, self.num_cat, self.latent_dim).view(-1, self.num_cat * self.latent_dim)
        # print (self.z_means)
        # if self.use_ib:
        #     posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        #     z = [d.rsample() for d in posteriors]
        #     self.z = torch.stack(z)
        # else:
        #     self.z = self.z_means

    def collect_kp(self, env, idx):
        data = env.get_kp_data()
        img_fr, choose_fr, cloud_fr, r_fr, t_fr, img_to, choose_to, cloud_to, r_to, t_to, mesh, faces, anchor, scale, cate, geodesic, curvature, depth_fr, mesh_orig, state_fr, quadric = data
        anchor = torch.Tensor([[[0., 0.6, 0.02]]]).to() / scale
        # print(f'img_fr.shape: {img_fr.shape}, choose_fr.shape: {choose_fr.shape}, cloud_fr.shape: {cloud_fr.shape}, r_fr.shape: {r_fr.shape}, t_fr.shape: {t_fr.shape}, mesh.shape: {mesh.shape}, anchor.shape: {anchor.shape}, scale.shape: {scale.shape}')
        img_fr, choose_fr, cloud_fr, r_fr, t_fr, img_to, choose_to, cloud_to, r_to, t_to, mesh, faces, anchor, scale, cate, geodesic, curvature, quadric = Variable(img_fr).cuda(torch.cuda.current_device()), \
                                                                                                                                                           Variable(choose_fr).cuda(torch.cuda.current_device()), \
                                                                                                                                                           Variable(cloud_fr).cuda(torch.cuda.current_device()), \
                                                                                                                                                           Variable(r_fr).cuda(torch.cuda.current_device()), \
                                                                                                                                                           Variable(t_fr).cuda(torch.cuda.current_device()), \
                                                                                                                                                           Variable(img_to).cuda(torch.cuda.current_device()), \
                                                                                                                                                           Variable(choose_to).cuda(torch.cuda.current_device()), \
                                                                                                                                                           Variable(cloud_to).cuda(torch.cuda.current_device()), \
                                                                                                                                                           Variable(r_to).cuda(torch.cuda.current_device()), \
                                                                                                                                                           Variable(t_to).cuda(torch.cuda.current_device()), \
                                                                                                                                                           Variable(mesh).cuda(torch.cuda.current_device()), \
                                                                                                                                                           Variable(faces).cuda(torch.cuda.current_device()), \
                                                                                                                                                           Variable(anchor).cuda(torch.cuda.current_device()), \
                                                                                                                                                           Variable(scale).cuda(torch.cuda.current_device()), \
                                                                                                                                                           Variable(cate).cuda(torch.cuda.current_device()), \
                                                                                                                                                           Variable(geodesic).cuda(torch.cuda.current_device()), \
                                                                                                                                                           Variable(curvature).cuda(torch.cuda.current_device()), \
                                                                                                                                                           Variable(quadric).cuda(torch.cuda.current_device())

        self.KeyNet.to(torch.cuda.current_device())
        Kp_fr, anc_fr, att_fr = self.KeyNet(img_fr.unsqueeze(0), choose_fr, cloud_fr, anchor, scale, cate, t_fr)
        Kp_to, anc_to, att_to = self.KeyNet(img_to.unsqueeze(0), choose_to, cloud_to, anchor, scale, cate, t_to)

        # Do we still need to update 6pack model?
        # loss, _, losses_dict = self.Kp_loss(Kp_fr, Kp_to, anc_fr, anc_to, att_fr, att_to, r_fr, t_fr, r_to, t_to, mesh, faces, scale, cate, geodesic, curvature, quadric)

        gt_Kp_fr = torch.bmm(Kp_fr - t_fr.unsqueeze(0), r_fr.unsqueeze(0)).contiguous()
        gt_Kp_to = torch.bmm(Kp_to - t_to.unsqueeze(0), r_to.unsqueeze(0)).contiguous()
        Kp = (gt_Kp_fr + gt_Kp_to) / 2
        out, kp_select, perm, edge_index = self.GNN(Kp, 0)

        ## [TODO]: remamber to do a permutation here, might be helpful
        # import pdb; pdb.set_trace()

        self.kp_list[idx] = kp_select
        self.tasks_z_list[idx] = out


    def get_kp_obs(self, obs, task_id):
        kp = self.kp_list[task_id]
        curr_t = obs['curr_hammer_pos']
        curr_quat = obs['curr_hammer_quat']

        kp = kp.squeeze().detach()
        q = torch.from_numpy(curr_quat[None, :]).to(kp.device)
        t = torch.from_numpy(curr_t[None, :]).to(kp.device)

        agent_kp = quaternion_apply(q, kp)
        agent_kp = agent_kp + t

        obs_gripper = torch.from_numpy(obs['curr_gripper_pos']).to(kp.device)

        agent_o = torch.cat((agent_kp, obs_gripper[None, :]), dim = 0)

        b, d = agent_o.shape
        agent_o = agent_o.view(b*d).unsqueeze(0)

        return agent_o.type(torch.cuda.FloatTensor)


    def get_obs_np(self, obs):
        init_hammer_pos = obs['init_hammer_pos']
        init_hammer_quat = obs['init_hammer_quat']
        curr_hammer_pos = obs['curr_hammer_pos']
        curr_hammer_quat = obs['curr_hammer_quat']
        curr_gripper_pos = obs['curr_gripper_pos']

        return np.hstack((init_hammer_pos, init_hammer_quat, curr_hammer_pos, curr_hammer_quat, curr_gripper_pos))


    def get_action(self, obs, deterministic=False):
        task_id = obs['task_id']
        kp_o = self.get_kp_obs(obs, task_id)
        z = self.tasks_z_list[task_id]

        ''' sample action from the policy, conditioned on the task embedding '''
        seq_z = ptu.FloatTensor()
        if self.glob:
            z = self.z
        if self.recurrent:
            seq_z = self.seq_z

        in_ = torch.cat([kp_o, z, seq_z], dim=1)
        # if self.recurrent:
        #     in_ = torch.cat([obs, z, seq_z], dim=1)
        # else:
        #     in_ = torch.cat([obs, z], dim=1)

        return self.policy.get_action(in_, deterministic=deterministic)

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def forward(self, obs, context, trajectories, indices_in_trajs, indices, do_inference, compute_for_next, is_next):
        ''' given context, get statistics under the current policy of a set of observations '''
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)

        task_z, seq_z = ptu.FloatTensor(), ptu.FloatTensor()

        # print("--------Inside inference network")
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        # start.record()

        # self.n_infer += 1
        if do_inference:
            if self.recurrent:
                assert type(self.recurrent_context_encoder) != type(None) and type(trajectories) != type(None) and type(
                    indices_in_trajs) != type(None)
                self.infer_sequence_posterior(trajectories, indices_in_trajs, compute_for_next=compute_for_next)
                # self.sample_sequence_z()
                # self.n_local += 1
            if self.glob:
                self.infer_posterior(context)
                # self.n_global += 1
                # self.sample_z()

        if self.recurrent:
            if is_next:
                seq_z = self.seq_z_next
            else:
                seq_z = self.seq_z

        if self.glob:
            task_z = self.z
            task_z = [z.repeat(b, 1) for z in task_z]
            task_z = torch.cat(task_z, dim=0)

        # end.record()
        # torch.cuda.synchronize()
        # print("do inference for z_seq: ", start.elapsed_time(end))
        # import pdb
        # pdb.set_trace()
        # task_z = task_z.unsqueeze(1).expand(t, b, task_z.size(-1))
        # task_z = task_z.view(t * b, task_z.size(-1))

        # run policy, get log probs and new actions
        # import pdb
        # pdb.set_trace()

        # start.record()
        task_z_pick = torch.stack([self.tasks_z_list[i] for i in indices], dim = 0)
        task_z_pick = task_z_pick.repeat(1, b, 1)
        task_z = task_z_pick.view(t * b, -1)

        in_ = torch.cat([obs, task_z.detach(), seq_z.detach()], dim=1)

        policy_outputs = self.policy(in_, reparameterize=True,
                                     return_log_prob=True)  # !! add flag, we donot need reparameterize for discrete action space, check

        # end.record()
        # torch.cuda.synchronize()
        # print("policy network forward: ", start.elapsed_time(end))
        # print("--------Outside inference network")

        return policy_outputs, task_z, seq_z

    def log_diagnostics(self, eval_statistics):  # !! modify if you have time
        '''
        adds logging data about encodings to eval_statistics
        '''

        if self.latent_dim > 0:
            z_mean = ptu.get_numpy(self.z_means[0])
            for i in range(len(z_mean)):
                eval_statistics['Z mean disc eval %d' % i] = z_mean[i]
        '''
        if self.cont_latent_dim > 0:
            z_mean = np.mean(np.abs(ptu.get_numpy(self.z_c_means[0])))
            z_sig = np.mean(ptu.get_numpy(self.z_c_vars[0]))
            eval_statistics['Z mean cont eval'] = z_mean
            eval_statistics['Z variance cont eval'] = z_sig
        '''
    @property
    def networks(self):
        network_list = []
        if self.glob:
            network_list.append(self.context_encoder)
        network_list.append(self.policy)
        if self.recurrent:
            network_list.append(self.recurrent_context_encoder)
        return network_list
        # if not self.recurrent:
        #     return [self.context_encoder, self.policy]
        # else:
        #     return [self.context_encoder, self.policy, self.recurrent_context_encoder]

    def infer_sequence_posterior(self, trajectories, indices_in_trajs, compute_for_next):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        # if ff:
        #     pdb.set_trace()
        # print (context.size())
        # print (context)
        num_tasks, traj_batch, eps_len, input_dim = trajectories.size()
        self.clear_sequence_z(num_tasks=num_tasks, batch_size=traj_batch * indices_in_trajs.size(2),
                              traj_batch_size=traj_batch)
        if self.rnn_sample == 'full':
            params = self.recurrent_context_encoder(trajectories.view(-1, eps_len, input_dim))
        elif self.rnn_sample == 'full_wo_sampling':
            params = self.recurrent_context_encoder(trajectories.view(-1, eps_len, input_dim))
        elif self.rnn_sample == 'single_sampling':
            traj_ranges = [i for i in range(eps_len) if i % self.temp_res == (self.temp_res - 1)]
            tmp_trajectories = trajectories[:, :, traj_ranges, :]
            params = self.recurrent_context_encoder(tmp_trajectories.view(-1, len(traj_ranges), input_dim))
            eps_len = len(traj_ranges)
        elif self.rnn_sample == 'batch_sampling':
            max_len = int(eps_len // self.temp_res * self.temp_res)
            tmp_trajectories = trajectories[:, :, :max_len, :]
            tmp_trajectories = tmp_trajectories.view(num_tasks * traj_batch, max_len // self.temp_res,
                                                     self.temp_res * input_dim)
            params = self.recurrent_context_encoder(tmp_trajectories)
            eps_len = max_len // self.temp_res

        if self.rnn_sample == 'full':
            if compute_for_next:
                indices_in_trajs_next = indices_in_trajs + 1
        else:
            if compute_for_next:
                indices_in_trajs_next = (indices_in_trajs + 1) // self.temp_res
            indices_in_trajs = indices_in_trajs // self.temp_res

        if self.vrnn_constraint == 'logitnormal':
            params = params.view(num_tasks, traj_batch, eps_len,
                                 self.r_cont_dim * 2 + self.r_n_cat * self.r_cat_dim + self.r_n_dir * self.r_dir_dim * 2)
        else:
            params = params.view(num_tasks, traj_batch, eps_len,
                                 self.r_cont_dim * 2 + self.r_n_cat * self.r_cat_dim + self.r_n_dir * self.r_dir_dim)

        if self.rnn_sample == 'full_wo_sampling':
            traj_ranges = [i for i in range(eps_len) if i % self.temp_res == (self.temp_res - 1)]
            params = params[:, :, traj_ranges, :]

        batch_per_traj = indices_in_trajs.size(2)
        params = torch.cat([self.seq_z_prior.expand(num_tasks, traj_batch, 1, params.size(3)), params], dim=2)

        if self.r_cat_dim > 0:
            params_disc = params[..., :self.r_n_cat * self.r_cat_dim]
            seq_z_cat = torch.gather(params_disc, 2,
                                     indices_in_trajs.unsqueeze(-1).expand(num_tasks, traj_batch, batch_per_traj,
                                                                           self.r_n_cat * self.r_cat_dim))
            self.seq_z_cat = F.softmax(
                seq_z_cat.view(num_tasks * traj_batch * batch_per_traj * self.r_n_cat, self.r_cat_dim), dim=-1)
            if compute_for_next:
                seq_z_next_cat = torch.gather(params_disc, 2,
                                              indices_in_trajs_next.unsqueeze(-1).expand(num_tasks, traj_batch,
                                                                                         batch_per_traj,
                                                                                         self.r_n_cat * self.r_cat_dim))
                self.seq_z_next_cat = F.softmax(
                    seq_z_next_cat.view(num_tasks * traj_batch * batch_per_traj * self.r_n_cat, self.r_cat_dim), dim=-1)
            else:
                self.seq_z_next_cat = None

        if self.r_cont_dim > 0:
            params_cont = params[...,
                          self.r_n_cat * self.r_cat_dim: self.r_n_cat * self.r_cat_dim + 2 * self.r_cont_dim]
            mu_c = params_cont[..., :self.r_cont_dim]
            sigma_squared_c = F.softplus(params_cont[..., self.r_cont_dim:])
            seq_z_cont_mean = torch.gather(mu_c, 2,
                                           indices_in_trajs.unsqueeze(-1).expand(num_tasks, traj_batch, batch_per_traj,
                                                                                 self.r_cont_dim))
            seq_z_cont_var = torch.gather(sigma_squared_c, 2,
                                          indices_in_trajs.unsqueeze(-1).expand(num_tasks, traj_batch, batch_per_traj,
                                                                                self.r_cont_dim))
            self.seq_z_cont_mean = seq_z_cont_mean.view(num_tasks * traj_batch * batch_per_traj, self.r_cont_dim)
            self.seq_z_cont_var = seq_z_cont_var.view(num_tasks * traj_batch * batch_per_traj, self.r_cont_dim)
            if compute_for_next:
                seq_z_next_cont_mean = torch.gather(mu_c, 2,
                                                    indices_in_trajs_next.unsqueeze(-1).expand(num_tasks, traj_batch,
                                                                                               batch_per_traj,
                                                                                               self.r_cont_dim))
                seq_z_next_cont_var = torch.gather(sigma_squared_c, 2,
                                                   indices_in_trajs_next.unsqueeze(-1).expand(num_tasks, traj_batch,
                                                                                              batch_per_traj,
                                                                                              self.r_cont_dim))
                self.seq_z_next_cont_mean = seq_z_next_cont_mean.view(num_tasks * traj_batch * batch_per_traj,
                                                                      self.r_cont_dim)
                self.seq_z_next_cont_var = seq_z_next_cont_var.view(num_tasks * traj_batch * batch_per_traj,
                                                                    self.r_cont_dim)
            else:
                self.seq_z_next_cont_var = None
                self.seq_z_next_cont_mean = None

        if self.r_dir_dim > 0 and self.vrnn_constraint == 'logitnormal':
            params_dir = params[..., self.r_n_cat * self.r_cat_dim + self.r_cont_dim * 2:]
            mu_d = params_dir[..., :self.r_n_dir * self.r_dir_dim]
            sigma_squared_d = F.softplus(params_dir[..., self.r_dir_dim * self.r_n_dir:])
            seq_z_dir_mean = torch.gather(mu_d, 2,
                                          indices_in_trajs.unsqueeze(-1).expand(num_tasks, traj_batch, batch_per_traj,
                                                                                self.r_n_dir * self.r_dir_dim))
            seq_z_dir_var = torch.gather(sigma_squared_d, 2,
                                         indices_in_trajs.unsqueeze(-1).expand(num_tasks, traj_batch, batch_per_traj,
                                                                               self.r_n_dir * self.r_dir_dim))
            self.seq_z_dir_mean = seq_z_dir_mean.view(num_tasks * traj_batch * batch_per_traj * self.r_n_dir,
                                                      self.r_dir_dim)
            self.seq_z_dir_var = seq_z_dir_var.view(num_tasks * traj_batch * batch_per_traj * self.r_n_dir,
                                                    self.r_dir_dim)
            if compute_for_next:
                seq_z_next_dir_mean = torch.gather(mu_d, 2,
                                                   indices_in_trajs_next.unsqueeze(-1).expand(num_tasks, traj_batch,
                                                                                              batch_per_traj,
                                                                                              self.r_n_dir * self.r_dir_dim))
                seq_z_next_dir_var = torch.gather(sigma_squared_d, 2,
                                                  indices_in_trajs_next.unsqueeze(-1).expand(num_tasks, traj_batch,
                                                                                             batch_per_traj,
                                                                                             self.r_n_dir * self.r_dir_dim))
                self.seq_z_next_dir_mean = seq_z_next_dir_mean.view(
                    num_tasks * traj_batch * batch_per_traj * self.r_n_dir, self.r_dir_dim)
                self.seq_z_next_dir_var = seq_z_next_dir_var.view(
                    num_tasks * traj_batch * batch_per_traj * self.r_n_dir, self.r_dir_dim)
            else:
                self.seq_z_next_dir_mean = None
                self.seq_z_next_dir_var = None

        if self.r_dir_dim > 0 and self.vrnn_constraint == 'dirichlet':
            params_dir = params[..., self.r_n_cat * self.r_cat_dim + self.r_cont_dim * 2:]
            seq_z_dir = torch.gather(params_dir, 2,
                                     indices_in_trajs.unsqueeze(-1).expand(num_tasks, traj_batch, batch_per_traj,
                                                                           self.r_n_dir * self.r_dir_dim))
            self.seq_z_dir = F.softplus(
                seq_z_dir.view(num_tasks * traj_batch * batch_per_traj * self.r_n_dir, self.r_dir_dim))
            if compute_for_next:
                seq_z_next_dir = torch.gather(params_dir, 2,
                                              indices_in_trajs_next.unsqueeze(-1).expand(num_tasks, traj_batch,
                                                                                         batch_per_traj,
                                                                                         self.r_n_dir * self.r_dir_dim))
                self.seq_z_next_dir = F.softplus(
                    seq_z_next_dir.view(num_tasks * traj_batch * batch_per_traj * self.r_n_dir, self.r_dir_dim))
            else:
                self.seq_z_next = None

        self.sample_sequence_z(compute_for_next)

    def sample_sequence_z(self, compute_for_next=False):
        z, z_c, z_d = ptu.FloatTensor(), ptu.FloatTensor(), ptu.FloatTensor()
        if self.r_cat_dim > 0:
            gumbel = torch.distributions.Gumbel(ptu.FloatTensor([0]), ptu.FloatTensor([1.0])).sample(
                self.seq_z_cat.size()).squeeze(-1)
            log_z = torch.log(self.seq_z_cat + eps)
            logit = (log_z + gumbel) / self.temperature
            z = F.softmax(logit, dim=1).view(-1, self.r_n_cat * self.r_cat_dim)
        if self.r_cont_dim > 0:
            normal = torch.distributions.Normal(ptu.FloatTensor([0.]), ptu.FloatTensor([1.])).sample(
                self.seq_z_cont_mean.size()).squeeze(-1)
            z_c = self.seq_z_cont_mean + torch.sqrt(self.seq_z_cont_var) * normal
        if self.r_dir_dim > 0:
            if self.vrnn_constraint == 'dirichlet':
                z_d = torch.distributions.Dirichlet(self.seq_z_dir).rsample().view(-1, self.r_n_dir * self.r_dir_dim)
            elif self.vrnn_constraint == 'logitnormal':
                normal = torch.distributions.Normal(ptu.FloatTensor([0.]), ptu.FloatTensor([1.])).sample(
                    self.seq_z_dir_mean.size()).squeeze(-1)
                z_d = F.softmax(self.seq_z_dir_mean + torch.sqrt(self.seq_z_dir_var) * normal, dim=-1).view(-1,
                                                                                                            self.r_n_dir * self.r_dir_dim)

        self.seq_z = torch.cat([z, z_c, z_d], dim=-1)

        if compute_for_next:
            z, z_c, z_d = ptu.FloatTensor(), ptu.FloatTensor(), ptu.FloatTensor()
            if self.r_cat_dim > 0:
                gumbel = torch.distributions.Gumbel(ptu.FloatTensor([0]), ptu.FloatTensor([1.0])).sample(
                    self.seq_z_next_cat.size()).squeeze(-1)
                log_z = torch.log(self.seq_z_next_cat + eps)
                logit = (log_z + gumbel) / self.temperature
                z = F.softmax(logit, dim=1).view(-1, self.r_n_cat * self.r_cat_dim)
            if self.r_cont_dim > 0:
                normal = torch.distributions.Normal(ptu.FloatTensor([0.]), ptu.FloatTensor([1.])).sample(
                    self.seq_z_next_cont_mean.size()).squeeze(-1)
                z_c = self.seq_z_next_cont_mean + torch.sqrt(self.seq_z_next_cont_var) * normal
            if self.r_dir_dim > 0:
                if self.vrnn_constraint == 'dirichlet':
                    z_d = torch.distributions.Dirichlet(self.seq_z_next_dir).rsample().view(-1,
                                                                                            self.r_n_dir * self.r_dir_dim)
                elif self.vrnn_constraint == 'logitnormal':
                    normal = torch.distributions.Normal(ptu.FloatTensor([0.]), ptu.FloatTensor([1.])).sample(
                        self.seq_z_next_dir_mean.size()).squeeze(-1)
                    z_d = F.softmax(self.seq_z_next_dir_mean + torch.sqrt(self.seq_z_next_dir_var) * normal,
                                    dim=-1).view(-1, self.r_n_dir * self.r_dir_dim)

            self.seq_z_next = torch.cat([z, z_c, z_d], dim=-1)
        else:
            self.seq_z_next = None

    def infer_step_posterior(self, step, resample):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        # if ff:
        #     pdb.set_trace()
        # print (context.size())
        # print (context)
        num_tasks = 1
        traj_batch = step.shape[0]
        # , eps_len, input_dim = trajectories.size()
        params = self.recurrent_context_encoder(step.view(num_tasks, traj_batch, -1))
        if resample:
            if self.vrnn_constraint == 'logitnormal':
                params = params.view(num_tasks, traj_batch,
                                     self.r_cont_dim * 2 + self.r_n_cat * self.r_cat_dim + self.r_n_dir * self.r_dir_dim * 2)
            else:
                params = params.view(num_tasks, traj_batch,
                                     self.r_cont_dim * 2 + self.r_n_cat * self.r_cat_dim + self.r_n_dir * self.r_dir_dim)

            if self.r_cat_dim > 0:
                # params_disc = params[..., :self.r_n_cat * self.r_cat_dim]
                seq_z_cat = params[..., :self.r_n_cat * self.r_cat_dim]
                self.seq_z_cat = F.softmax(seq_z_cat.view(num_tasks * traj_batch * self.r_n_cat, self.r_cat_dim),
                                           dim=-1)

            if self.r_cont_dim > 0:
                params_cont = params[...,
                              self.r_n_cat * self.r_cat_dim:self.r_n_cat * self.r_cat_dim + 2 * self.r_cont_dim]
                seq_z_cont_mean = params_cont[..., :self.r_cont_dim]
                seq_z_cont_var = F.softplus(params_cont[..., self.r_cont_dim:])
                self.seq_z_cont_mean = seq_z_cont_mean.view(num_tasks * traj_batch, self.r_cont_dim)
                self.seq_z_cont_var = seq_z_cont_var.view(num_tasks * traj_batch, self.r_cont_dim)

            if self.r_dir_dim > 0 and self.vrnn_constraint == 'logitnormal':
                params_dir = params[..., self.r_n_cat * self.r_cat_dim + self.r_cont_dim * 2:]
                seq_z_dir_mean = params_dir[..., :self.r_n_dir * self.r_dir_dim]
                seq_z_dir_var = F.softplus(params_dir[..., self.r_dir_dim * self.r_n_dir:])
                self.seq_z_dir_mean = seq_z_dir_mean.view(num_tasks * traj_batch * self.r_n_dir, self.r_dir_dim)
                self.seq_z_dir_var = seq_z_dir_var.view(num_tasks * traj_batch * self.r_n_dir, self.r_dir_dim)

            if self.r_dir_dim > 0 and self.vrnn_constraint == 'dirichlet':
                seq_z_dir = params[..., self.r_n_cat * self.r_cat_dim + self.r_cont_dim * 2:]
                self.seq_z_dir = F.softplus(seq_z_dir.view(num_tasks * traj_batch * self.r_n_dir, self.r_dir_dim))

            self.sample_sequence_z()

    def clear_sequence_z(self, num_tasks=1, batch_size=1,
                         traj_batch_size=1):  # ! check when it happens, because it looks fishy for batch size part
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        # self.hn = ptu.ones(num_tasks, ) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!update here
        assert self.recurrent_context_encoder != None
        if self.r_cat_dim > 0:
            self.seq_z_cat = ptu.ones(num_tasks * batch_size * self.r_n_cat, self.r_cat_dim) / self.r_cat_dim
            self.seq_z_next_cat = None
        if self.r_cont_dim > 0:
            self.seq_z_cont_mean = ptu.zeros(num_tasks * batch_size, self.r_cont_dim)
            self.seq_z_cont_var = ptu.ones(num_tasks * batch_size, self.r_cont_dim)
            self.seq_z_next_cont_mean = None
            self.seq_z_next_cont_var = None
        if self.r_dir_dim > 0:
            if self.vrnn_constraint == 'logitnormal':
                self.seq_z_dir_mean = ptu.zeros(num_tasks * batch_size * self.r_n_dir, self.r_dir_dim)
                self.seq_z_dir_var = ptu.ones(num_tasks * batch_size * self.r_n_dir, self.r_dir_dim) * self.r_var
                self.seq_z_next_dir_mean = None
                self.seq_z_next_dir_var = None
            elif self.vrnn_constraint == 'dirichlet':
                self.seq_z_dir = ptu.ones(num_tasks * batch_size * self.r_n_dir, self.r_dir_dim) * self.r_alpha
                self.seq_z_next_dir = None

        self.sample_sequence_z()
        self.recurrent_context_encoder.reset(num_tasks * traj_batch_size)
