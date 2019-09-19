import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import math
import scipy.stats as stats
import lsdr.envs.environment_sampler as env_sampler
from enum import IntEnum


############################
# Optimization Loss Opt
############################
class Objectives(IntEnum):
    REWARDS = 1
    KL_OPT = 2
    REW_AND_KL = 3


def reward_function(x):
    return np.exp(-(x-20)**2)

def reward_function_v2(x):

    return np.sin(np.sqrt(x**2))

def calculate_reward(x):

    return reward_function(x)

def setup_distributions():

    ##############################
    # Initial distribution configs
    ##############################
    test_params = [
        np.array([-30.0, 50.0])
    ]

    # This can be modified for the initial distributions
    # to be different.
    ranges = np.asarray(test_params)
    mean = ranges.mean(-1)
    covar = (((ranges[:, 1] - ranges[:, 0])**2.0) / 12.0) * np.eye(
        ranges.shape[0])
    mu_train, L_train = mean, np.linalg.cholesky(covar)

    dist_params = [mu_train, L_train]


    sampler = env_sampler.init_env_sampler(
        'hopper',
        seed=0,
        experiment_id='test_kl_div_loss_0',
        init_dist_params=dist_params,
        dist_type='gaussian',
        test_dist_params=None)

    ############################
    # Train Distribution
    ############################
    p_train = sampler.train_dist

    ############################
    # Test Distribution
    ############################

    ranges = np.asarray(test_params)
    mean = ranges.mean(-1)
    covar = (((ranges[:, 1] - ranges[:, 0])**2.0) / 12.0) * np.eye(
        ranges.shape[0])
    mu_test, L_test = mean, np.linalg.cholesky(covar)

    mu_test = torch.tensor(mu_test)
    L_test = torch.tensor(L_test)

    mu_test = mu_test.float().detach().requires_grad_(False)
    L_test = L_test.float().detach().requires_grad_(False)
    p_test = torch.distributions.MultivariateNormal(mu_test,
                                                    scale_tril=L_test)

    train_mean = p_train.mean.detach()
    train_std = (p_train._unbroadcasted_scale_tril).diag().detach()
    test_mean = p_test.mean.detach()
    test_std = (p_test._unbroadcasted_scale_tril).diag().detach()

    print('Initial Distributions')
    print('Train Distribution Mean ', train_mean)
    print('Train Distribution STD ', train_std)
    print('Test Distribution Mean ', test_mean)
    print('Test Distribution STD ', test_std)

    ############################
    # Plot Initial Distribution
    ############################

    plot_distrs(train_mean, train_std,
                test_mean, test_std,
                plot_name='initial_train_distr')

    return sampler, p_train, p_test


def plot_distrs(train_mean, train_var,
                test_mean, test_var,
                plot_name='distributions'):

    plt.figure()
    mu = train_mean
    variance = train_var
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), color='green',
             label='$p_{\phi}(z)$',
             linestyle='-.')
    mu = test_mean
    variance = test_var
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), color='red', label='$p(z)$')

    rew_func_range = np.arange(-20, 50, 1)
    plt.plot(rew_func_range, calculate_reward(rew_func_range),
             color='orange',
             label='$R(\Theta, z)$')

    plt.legend(loc='upper left')

    res_dir = 'grad_analysis'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    plotname = res_dir + '/' + plot_name + '.png'
    plt.savefig(plotname)


def optimize_distribution(sampler, p_train, p_test, objective_opt):
    epochs, n_samples = 10000, 1000

    alpha = 1e-5

    opt = torch.optim.Adam(sampler.params, 1e-2)

    mu_grads = []
    var_grads = []

    def store_mu_grad_rew(grad):
        mu_grads.append(np.copy(grad))

    def store_tril_grad_rew(grad):
        var_grads.append(np.copy(grad))

    for _ in range(epochs):
        opt.zero_grad()

        ####################
        # Sample from p_test
        ####################
        z = p_test.sample(torch.Size([n_samples]))
        contexts = p_train.sample(torch.Size([n_samples]))

        ################
        # Eval Log probs
        ################
        log_p_train = p_train.log_prob(z)
        log_p_test = p_test.log_prob(z)

        ################
        # Calculate KL
        ################
        kl_samples = log_p_test - log_p_train
        kl_loss = kl_samples.mean(0)

        #######################
        # Calculate Reward term
        #######################
        log_probs_context = p_train.log_prob(contexts)
        reward_loss = (calculate_reward(contexts) * log_probs_context).mean(0)

        if objective_opt == Objectives.REWARDS:
            # For this to converge to the reward function,
            # need to change `z` sampling to be from train
            # distribution.
            total_loss = - reward_loss

        elif objective_opt == Objectives.KL_OPT:
            total_loss = kl_loss

        elif objective_opt == Objectives.REW_AND_KL:
            total_loss = (-(reward_loss) + (alpha*kl_loss))

        else:
            raise ValueError('Invalid op')

        total_loss.mean().backward()
        opt.step()

    train_mean = p_train.mean.detach()
    train_std = (p_train._unbroadcasted_scale_tril).diag().detach()
    test_mean = p_test.mean.detach()
    test_std = (p_test._unbroadcasted_scale_tril).diag().detach()

    print('Updated Distributions')
    print('######################')
    print('Train Distribution Mean ', train_mean)
    print('Train Distribution STD ', train_std)
    print('Test Distribution Mean ', test_mean)
    print('Test Distribution STD ', test_std)

    plot_distrs(train_mean, train_std,
                test_mean, test_std,
                plot_name='final_distributions')


if __name__ == '__main__':
    sampler, p_train, p_test = setup_distributions()

    # objective_opt = Objectives.REWARDS
    # objective_opt = Objectives.KL_OPT
    objective_opt = Objectives.REW_AND_KL

    optimize_distribution(sampler,
                          p_train,
                          p_test,
                          objective_opt)
