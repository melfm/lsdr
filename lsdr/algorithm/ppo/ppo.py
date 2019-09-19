import numpy as np
import os
import tensorflow as tf
import time
import pickle

import gym
import torch

import lsdr.algorithm.ppo.core as core
from enum import IntEnum
from lsdr.algorithm.ppo.normalizer import Normalizer
from lsdr.utils.logx import EpochLogger, Logger
from lsdr.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from lsdr.utils.mpi_tools import (mpi_avg, proc_id, mpi_statistics_scalar,
                                  num_procs)
from lsdr.envs.environment_sampler import sample_env


class Objectives(IntEnum):
    REWARDS = 1
    IMPROVEMENT = 2


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation
    (GAE-Lambda) for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim,
                 size, gamma=0.99, lam=0.95,
                 context_dim=None):
        self.obs_buf = np.zeros(
            core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(
            core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.mask = np.ones(size, dtype=np.float32)
        self.mask /= self.mask.sum()
        self.path_slices = []
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

        if context_dim is not None:
            self.context_buf = np.zeros([size, context_dim],
                                        dtype=np.float32)

    def store(self, obs, act, rew, val, logp, context=None):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        if context is not None:
            if self.ptr >= self.max_size:
                # Skip storing - this happens with `ctx_val_buf`
                return
        else:

            # Otherwise buffer has to have room so you can store
            assert self.ptr < self.max_size

        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        if context is not None:
            self.context_buf[self.ptr] = context
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        self.path_slices.append(path_slice)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(
            deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value
        # function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def update_mask(self, eps=0.1, empirical=False, smoothing=1e-1):
        """Compute the mask for EPOpt.
        """
        if empirical:
            R = np.array([self.rew_buf[s].sum() for s in self.path_slices])
        else:
            R = np.array([self.ret_buf[s][0] for s in self.path_slices])

        # EPOpt uses the quantile calculation to only pay attention to the
        # epsilon worst trajectories
        cutoff = np.percentile(R, eps * 100)
        mask = R <= cutoff
        # Use the weights to select the worst trajectories
        weights = np.array(mask, dtype=np.float32)
        # Smoothing allows for better performing trajectories to also
        # contribute to the gradients, but by a smaller amount than the
        # worst ones
        if smoothing > 0:
            weights[mask] -= smoothing
            weights[~mask] += smoothing

        for i, s in enumerate(self.path_slices):
            self.mask[s] = weights[i]

        # Normalize the mask
        self.mask /= self.mask.sum()

    def get(self, epopt_sampling=False):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        if not epopt_sampling:
            # buffer has to be full before you can get
            assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        self.path_slices = []
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [
            self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf,
            self.logp_buf
        ]


class PPO:
    def __init__(self,
                 context_sampler,
                 actor_critic=core.mlp_actor_critic,
                 ac_kwargs=dict(),
                 seed=0,
                 steps_per_epoch=4000,
                 epochs=50,
                 gamma=0.99,
                 clip_ratio=0.2,
                 pi_lr=3e-4,
                 vf_lr=1e-3,
                 train_pi_iters=80,
                 train_v_iters=80,
                 lam=0.97,
                 max_ep_len=1000,
                 target_kl=0.01,
                 logger_kwargs=dict(),
                 save_freq=50,
                 render=False,
                 cat_context=False,
                 normalize_ctx=False,
                 normalize_obs=False,
                 context_samples=10,
                 store_context_buffer=True,
                 alpha=1.0,
                 eval_eps=1,
                 learn_distr=True,
                 reuse_ctx_samples=False,
                 dist_param_idxs=None,
                 distr_learning_iters=10,
                 enable_epopt=False,
                 epopt_eps=0.1,
                 epopt_trajs=100,
                 epopt_warm_start_epochs=100,
                 epopt_warm_start_eps=1.0,
                 console_out=True,
                 use_kl_regularizer=False,
                 dr_objective=Objectives.REWARDS,
                 whiten_l_dr=True,
                 DR_learn_sample_from='test',
                 use_IS_weights=False,
                 **kwargs):
        """Proximal Policy Optimization (by clipping),
        with early stopping based on approximate KL

        Args:
            context_sampler: Wrapper to manage context distribution and
                sample gym environment.

            actor_critic: A function which takes in placeholder symbols
                for state, ``x_ph``, and action, ``a_ph``, and returns the main
                outputs from the agent's Tensorflow computation graph:

                ===========  ================  ================================
                Symbol       Shape             Description
                ===========  ================  ================================
                ``pi``       (batch, act_dim)  Samples actions from policy
                                               given the state.
                ``logp``     (batch,)          Gives log probability, according
                                               to the policy, of taking actions
                                               ``a_ph`` in states ``x_ph``.
                ``logp_pi``  (batch,)          Gives log probability, according
                                               to the policy, of the action
                                               sampled by ``pi``.
                ``v``        (batch,)          Gives the value estimate for
                                               states in ``x_ph``. (Critical:
                                               make sure to flatten this!)
                ===========  ================  ================================

            ac_kwargs (dict): Any kwargs appropriate for the actor_critic
                function you provided to PPO.

            store_context_buffer (bool): Store trajectories labelled with context
                locally for performing context inference offline.

            seed (int): Seed for random number generators.

            steps_per_epoch (int): Number of steps of interaction (state-action
                pairs) for the agent and the environment in each epoch.

            epochs (int): Number of epochs of interaction (equivalent to
                number of policy updates) to perform.

            gamma (float): Discount factor. (Always between 0 and 1.)

            clip_ratio (float): Hyperparameter for clipping in the policy
                objective. Roughly: how far can the new policy go from the old
                policy while still profiting (improving the objective function)
                The new policy can still go farther than the clip_ratio says,
                but it doesn't help on the objective anymore. (Usually small,
                0.1 to 0.3.)

            pi_lr (float): Learning rate for policy optimizer.

            vf_lr (float): Learning rate for value function optimizer.

            train_pi_iters (int): Maximum number of gradient descent steps to
                take on policy loss per epoch. (Early stopping may cause
                optimizer to take fewer than this.)

            train_v_iters (int): Number of gradient descent steps to take on
                value function per epoch.

            lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
                close to 1.)

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            target_kl (float): Roughly what KL divergence we think is
                appropriate between new and old policies after an update.
                This will get used for early stopping. (Usually small, 0.01 or
                0.05.)
            reuse_ctx_samples (bool): This option enables re-using contexts
                from the training period in order to reduce the number of
                required context resampling during the distribution learning.

            normalize_ctx (bool): Normalize the contexts to a value between
                0 - 1.

            normalize_obs (bool) : Normalize the observation data. If `normalize_ctx`
                is enabled, the concatenated observation with already-normalized
                contexts will be normalized.

            enable_epopt (bool): Switch to EPOpt CVaR objective instead.

            epopt_eps (float): Number between 0.0 and 1.0 representing the
                percentage of trajectories from the buffer to be used for
                training the policy. when epopt_eps = 1.0, this is equivalent
                to using regular ppo. When epopt_eps < 1.0, the epopt_eps*100%
                trajectories with lowest cumulative returns will be selected
                for training.

            epopt_trajs (int): Number of trajectories used to evaluate the
                worst epsilon trajectories. This adds additional trajectories
                and increases the buffer size until this number is satisfied.

            epopt_warm_start_epochs (int): Epoch at which to start EPOpt
                sampling.

            epopt_warm_start_eps (float): Initial epsilon for during of warm
                start period.

            logger_kwargs (dict): Keyword args for EpochLogger.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.

            use_kl_regularizer(bool): if True, use kl regularizer otherwise uses
                 entropy

            dr_objective(int): Select the score used to decide which contexts to
                make more likely (the domain randomization objective).
                1: Uses empirical returns
                2: Uses emprical relative improvement over the old policy

            whiten_l_dr: Whether to standardize the  domain randomization objective
                scores. Standardizing is done by computing the running mean and
                variance of the dr objective, using them to center and rescale the
                scores to have mean 0 and variance of 1.

            DR_learn_sample_from: allowed values ['train', 'test']. Which
                samples to use for distribution learning

            use_IS_weights: whther to use importance sampling weights
        """

        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.lam = lam
        self.max_ep_len = max_ep_len
        self.target_kl = target_kl

        self.context_sampler = context_sampler
        self.learn_distr = learn_distr
        self.cat_context = cat_context
        self.normalize_ctx = normalize_ctx
        self.normalize_obs = normalize_obs
        self.eval_eps = eval_eps
        self.reuse_ctx_samples = reuse_ctx_samples
        self.context_samples = context_samples
        self.store_context_buffer = store_context_buffer
        self.alpha = alpha
        self.dist_param_idxs = dist_param_idxs
        self.distr_learning_iters = distr_learning_iters
        self.use_kl_regularizer = use_kl_regularizer
        self.dr_objective = dr_objective
        self.whiten_l_dr = whiten_l_dr
        self.DR_learn_sample_from = DR_learn_sample_from
        self.use_IS_weights = use_IS_weights

        self.ctx_counter = 0
        self.R_mean, self.R_std = 0.0, 1.0
        self.R_min = 0.0
        if self.normalize_ctx:
            # Get the min and max stat from the test distribution
            # as its typically a discrete distribution. It is assumed
            # that the train distribution is initialized to test distr.
            self.context_min = self.context_sampler.test_dist.lo
            self.context_max = self.context_sampler.test_dist.hi
            self.context_range = self.context_max - self.context_min
        if self.normalize_obs:
            # This will have to be initialized after the session
            self.obs_norm = None
        self.sess = None

        self.save_freq = save_freq
        self.logger_kwargs = logger_kwargs
        self.console_out = console_out
        self.render = render
        self.seed = seed

        # EPOpt setting
        self.enable_epopt = enable_epopt
        self.epopt_eps = epopt_eps
        self.epopt_trajs = epopt_trajs
        self.epopt_warm_start_epochs = epopt_warm_start_epochs
        self.epopt_warm_start_eps = epopt_warm_start_eps

        # Reset the graph
        tf.reset_default_graph()

        # Set random seed
        self.seed += 10000 * proc_id()
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)

        # Set up logging
        self.logger = EpochLogger(**self.logger_kwargs)
        local_params = locals().copy()
        local_params.pop('self', None)
        self.logger.save_config(local_params)
        self._setup_ctx_loggers()

        env_sample = sample_env(self.context_sampler)
        self.obs_dim = self.context_sampler.get_observation_space(
            self.cat_context, full_dim=True)

        if self.cat_context:
            _, context_dim = self.context_sampler.get_observation_space(
                self.cat_context, full_dim=False)
        else:
            context_dim = 0

        act_dim = env_sample.action_space.shape

        self.viewers = {}
        self.init_viewers(env_sample, self.viewers)

        # Share information about action space with policy architecture
        ac_kwargs['action_space'] = env_sample.action_space

        # Inputs to computation graph
        self.x_ph, self.a_ph = core.placeholders_from_spaces(
            self.obs_dim, env_sample.action_space)
        self.adv_ph, self.ret_ph, self.logp_old_ph = core.placeholders(
            None, None, None)

        # EPOpt mask
        self.mask_ph = core.placeholder(None)

        # Main outputs from computation graph
        self.pi, self.logp, self.logp_pi, self.v = actor_critic(
            self.x_ph, self.a_ph, actor_scope='pi', critic_scope='v',
            **ac_kwargs)

        # Need all placeholders in *this* order later (to zip with data from
        # buffer)
        self.all_phs = [
            self.x_ph, self.a_ph, self.adv_ph, self.ret_ph, self.logp_old_ph
        ]

        # Every step, get: action, value, and logprob
        self.get_action_ops = [self.pi, self.v, self.logp_pi]

        if self.dr_objective == Objectives.IMPROVEMENT:
            # create copy of actor critic for previous policy
            self.prev_ac = actor_critic(
                self.x_ph, self.a_ph, actor_scope='pi_old',
                critic_scope='v_old', **ac_kwargs)

            # policy parameters
            self.pi_params = tf.trainable_variables('pi/')
            self.pi_old_params = tf.trainable_variables('pi_old/')

            # old policy ops
            # pi_old, v_old, log_pi_old
            self.old_action_ops = [
                self.prev_ac[0], self.prev_ac[3], self.prev_ac[2]]

        # Experience buffer
        self.local_steps_per_epoch = int(self.steps_per_epoch / num_procs())
        buffer_size = self.local_steps_per_epoch
        if self.enable_epopt:
            # Expand to maximum size possible for EPOpt worst trajectory
            # sampling
            buffer_size *= self.epopt_trajs

        if self.store_context_buffer:
            self.buf = PPOBuffer(self.obs_dim, act_dim, buffer_size, self.gamma,
                                 self.lam, context_dim)
            self.ctx_val_buf = PPOBuffer(
                self.obs_dim,
                act_dim,
                buffer_size,
                self.gamma,
                self.lam,
                context_dim)
        else:
            self.buf = PPOBuffer(self.obs_dim, act_dim, buffer_size, self.gamma,
                                 self.lam)

        self.setup_ppo_objectives()

    def _setup_ctx_loggers(self):

        self.context_logger = Logger(
            output_dir=self.logger_kwargs['output_dir'],
            output_fname='contexts.txt')

        if self.learn_distr:
            self.distribution_logger = Logger(
                output_dir=self.logger_kwargs['output_dir'],
                output_fname='distributions.txt')

            self.debugger_logger = Logger(
                output_dir=self.logger_kwargs['output_dir'],
                output_fname='debugger.txt')

        if self.store_context_buffer:
            self.context_storage_path = self.logger_kwargs['output_dir'] + '/' +\
                'context_buffer'
            if not os.path.exists(self.context_storage_path):
                os.makedirs(self.context_storage_path)

    def setup_ppo_objectives(self):
        # PPO objectives
        # pi(a|s) / pi_old(a|s)
        ratio = tf.exp(self.logp - self.logp_old_ph)
        min_adv = tf.where(self.adv_ph > 0,
                           (1 + self.clip_ratio) * self.adv_ph,
                           (1 - self.clip_ratio) * self.adv_ph)

        if self.enable_epopt:
            # if the epopt epsilon is a valid percentage number
            # (less than one), this implements the CVaR objective
            self.pi_loss = -tf.reduce_sum(
                self.mask_ph * tf.minimum(ratio * self.adv_ph, min_adv))
            self.v_loss = tf.reduce_sum(
                self.mask_ph * (self.ret_ph - self.v)**2)

            # Info (useful to watch during learning)
            # a sample estimate for KL-divergence, easy to compute
            self.approx_kl = tf.reduce_sum(
                self.mask_ph * (self.logp_old_ph - self.logp))
            # a sample estimate for entropy, also easy to compute
            self.approx_ent = tf.reduce_sum(self.mask_ph * (-self.logp))
            self.clipped = tf.logical_or(ratio > (1 + self.clip_ratio),
                                         ratio < (1 - self.clip_ratio))
            self.clipfrac = tf.reduce_sum(
                self.mask_ph * tf.cast(self.clipped, tf.float32))
        else:

            self.pi_loss = -tf.reduce_mean(
                tf.minimum(ratio * self.adv_ph, min_adv))
            self.v_loss = tf.reduce_mean((self.ret_ph - self.v)**2)

            # Info (useful to watch during learning)
            # a sample estimate for KL-divergence, easy to compute
            self.approx_kl = tf.reduce_mean(self.logp_old_ph - self.logp)
            # a sample estimate for entropy, also easy to compute
            self.approx_ent = tf.reduce_mean(-self.logp)
            self.clipped = tf.logical_or(ratio > (1 + self.clip_ratio),
                                         ratio < (1 - self.clip_ratio))
            self.clipfrac = tf.reduce_mean(tf.cast(self.clipped, tf.float32))

        # Optimizers
        self.train_pi = MpiAdamOptimizer(learning_rate=self.pi_lr).minimize(
            self.pi_loss)
        self.train_v = MpiAdamOptimizer(learning_rate=self.vf_lr).minimize(
            self.v_loss)

    def update_policy(self, mask=None):
        inputs = {k: v for k, v in zip(self.all_phs,
                                       self.buf.get(self.enable_epopt))}
        if self.enable_epopt and mask is not None:
            inputs[self.mask_ph] = mask
        pi_l_old, v_l_old, ent = self.sess.run(
            [self.pi_loss, self.v_loss, self.approx_ent], feed_dict=inputs)

        if self.dr_objective == Objectives.IMPROVEMENT:
            # keep copy of old policy
            self.sess.run(
                [tf.assign(o, p) for (o, p)
                 in zip(self.pi_old_params, self.pi_params)])

        # Training
        for i in range(self.train_pi_iters):
            _, kl = self.sess.run([self.train_pi, self.approx_kl],
                                  feed_dict=inputs)
            kl = mpi_avg(kl)
            if kl > 1.5 * self.target_kl:
                if self.console_out:
                    self.logger.log(
                        'Early stopping at step %d due to reaching max kl.' %
                        i)
                break
        self.logger.store(StopIter=i)
        for _ in range(self.train_v_iters):
            self.sess.run(self.train_v, feed_dict=inputs)

        # Log changes from update
        pi_l_new, v_l_new, kl, cf = self.sess.run(
            [self.pi_loss, self.v_loss, self.approx_kl, self.clipfrac],
            feed_dict=inputs)
        self.logger.store(
            LossPi=pi_l_old,
            LossV=v_l_old,
            KL=kl,
            Entropy=ent,
            ClipFrac=cf,
            DeltaLossPi=(pi_l_new - pi_l_old),
            DeltaLossV=(v_l_new - v_l_old))

    def concat_context(self, obs, context):
        """Concatenate context to the observations.

        Args:
            obs: numpy array, observations
            context: torch.tensor, context vector.
        """
        if self.normalize_ctx:
            ctx_normalized = \
                (context - self.context_min) / self.context_range
            return np.append(obs, ctx_normalized)
        return np.append(obs, context)

    def get_norm_obs(self, obs):
        self.obs_norm.update(obs)
        return self.obs_norm.normalize(obs)

    def rollout_policy(self, epoch):
        """This does the buffer filling part of PPO.

        Args:
            epoch (int): Current epoch of training, used to store
                context buffer files.
        """

        sampled_contexts = []
        valid_contexts = []
        episode_returns = []

        #######################
        # Sample the first env
        #######################
        env = sample_env(self.context_sampler)
        self.update_viewers(env, self.viewers)
        valid_ctx = True
        self.ctx_counter += 1
        obs, reward, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        context = env.context
        if self.cat_context:
            obs = self.concat_context(obs, context)
        if self.normalize_obs:
            obs = self.get_norm_obs(obs)
        epoch_start_time = time.time()

        def generate_trajectory(env, obs, reward, done, context, valid_ctx,
                                ep_ret, ep_len, sampled_contexts,
                                valid_contexts, episode_returns):

            for step in range(self.local_steps_per_epoch):
                action, v_t, logp_t = self.sess.run(
                    self.get_action_ops,
                    feed_dict={self.x_ph: obs.reshape(1, -1)})

                # Save trajectory and log v_t
                if self.store_context_buffer:
                    self.buf.store(obs, action, reward, v_t, logp_t, context)
                else:
                    self.buf.store(obs, action, reward, v_t, logp_t)
                self.logger.store(VVals=v_t)

                try:
                    obs, reward, done, _ = env.step(action[0])
                    if self.cat_context:
                        obs = self.concat_context(obs, context)
                    if self.normalize_obs:
                        obs = self.get_norm_obs(obs)
                    ep_ret += reward
                    ep_len += 1

                    if self.render:
                        env.render()

                except Exception:
                    valid_ctx = False

                terminal = done or (ep_len == self.max_ep_len)
                if terminal or (step == self.local_steps_per_epoch -
                                1) or not valid_ctx:
                    if not (terminal) and self.console_out:
                        print('Warning: trajectory cut off by epoch at \
                                %d steps.' % ep_len)

                    episode_returns = self.terminate_trajectory(
                        obs, reward, done, valid_ctx, context, ep_ret, ep_len,
                        episode_returns, terminal)

                    sampled_contexts.append(env.context)
                    valid_contexts.append(valid_ctx)

                    # Try sampling a new environment until a valid env is found
                    env, context = self.context_sampler.sample(
                        return_ctxs=True)
                    while env is None:
                        # assume that the task is not solvable with the given
                        # context, thus it returns the lowest possible rewards
                        ep_ret = self.R_min
                        self.log_context(context, ep_ret)
                        episode_returns.append(ep_ret)
                        sampled_contexts.append(context)
                        valid_contexts.append(False)

                        # sample a new one
                        env, context = self.context_sampler.sample(
                            return_ctxs=True)

                    self.update_viewers(env, self.viewers)
                    valid_ctx = True
                    self.ctx_counter += 1
                    obs, reward, done, ep_ret, ep_len = env.reset(
                    ), 0, False, 0, 0
                    if self.cat_context:
                        context = env.context
                        obs = self.concat_context(obs, context)

            return env, sampled_contexts, valid_contexts, episode_returns

        if self.enable_epopt:
            n_trajs = 0
            while n_trajs < self.epopt_trajs:
                env, sampled_contexts, valid_contexts, episode_returns =\
                    generate_trajectory(env, obs, reward, done, context,
                                        valid_ctx,
                                        ep_ret, ep_len,
                                        sampled_contexts,
                                        valid_contexts, episode_returns)

                # At this point decide, whether you have enough eps%
                # trajectories
                trajs_ret = np.array(
                    [self.buf.ret_buf[s][0] for s in self.buf.path_slices])
                # EPOpt uses the quantile calculation to only pay attention to
                # the epsilon worst trajectories
                eps = 0.1
                cutoff = np.percentile(trajs_ret, eps * 100)
                n_trajs += np.where(trajs_ret < cutoff)[0].shape[0]

        else:
            env, sampled_contexts, valid_contexts, episode_returns =\
                generate_trajectory(env, obs, reward, done, context,
                                    valid_ctx,
                                    ep_ret, ep_len,
                                    sampled_contexts,
                                    valid_contexts, episode_returns)
            if self.store_context_buffer:
                # Store context buffer info to file
                context_buf_filename = self.context_storage_path + \
                    '/' + str(epoch) + '.pkl'
                with open(context_buf_filename, 'wb') as output:
                    # TODO: Change this to create a copy of object with only
                    # obs_buf, act_buf and context_buf data.
                    pickle.dump(self.buf, output, pickle.HIGHEST_PROTOCOL)

        return env, sampled_contexts, valid_contexts, \
            episode_returns, epoch_start_time

    def terminate_trajectory(self, obs, reward, done, valid_ctx, context,
                             ep_ret, ep_len, episode_returns, terminal):
        """Makes a call to self.buf.finish_path to tag the trajectory
        as finished. It then logs and returns the episode rewards.
        This function takes the `latest` observations, reward etc
        and depending on whether the context was solvable or not,
        stores the appropriate final reward.
        """
        if valid_ctx:
            # if trajectory didn't reach terminal state, bootstrap
            # value target
            last_val = reward if done else self.sess.run(
                self.v, feed_dict={self.x_ph: obs.reshape(1, -1)})
            ep_ret = ep_ret if done else ep_ret + \
                np.asscalar(last_val)
            self.buf.finish_path(last_val)
            episode_returns.append(ep_ret)
        else:
            episode_returns.append(self.R_min)

        if terminal:
            # only save EpRet / EpLen if trajectory finished
            self.logger.store(EpRet=ep_ret, EpLen=ep_len)
            self.log_context(context, ep_ret)

        return episode_returns

    def train_ppo(self):
        """Main training function for the policy and distribution learning.
        """

        #########################
        # Create the session
        #########################
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        with tf.variable_scope("normalizer"):
            self.obs_norm = Normalizer(self.obs_dim[0], sess=self.sess)

        self.sess.run(tf.global_variables_initializer())

        # Sync params across processes
        self.sess.run(sync_all_params())

        # Setup model saving
        self.logger.setup_tf_saver(
            self.sess,
            inputs={'x': self.x_ph},
            outputs={
                'pi': self.pi,
                'v': self.v
            })

        epoch_time = None
        start_time = time.time()

        torch.save(self.context_sampler,
                   self.logger_kwargs['output_dir'] + "/context_dist.pth")

        if self.learn_distr:
            if self.dist_param_idxs is None:
                dist_param_idxs = list(range(len(self.context_sampler.params)))
            cs_params = [
                self.context_sampler.params[idx] for idx in dist_param_idxs
            ]
            self.distr_optimizer = torch.optim.Adam(cs_params, 1e-3)
            self.distr_optimizer.zero_grad()

        for epoch in range(self.epochs):

            ########################
            # Step 1. Rollout policy
            ########################
            env, sampled_contexts, valid_contexts, \
                episode_returns, epoch_start_time = self.rollout_policy(epoch)

            ###########################################
            # Step 2. Update sim distribution
            # parameters. Evaluate to collect rewards,
            # optimize context distribution
            ##########################################
            if self.learn_distr:
                self.learn_sim_distribution(sampled_contexts, episode_returns,
                                            valid_contexts)

            # Save model
            if (epoch % self.save_freq == 0) or (epoch == self.epochs - 1):
                self.save_model(env)

            #############################
            # Step 3. Perform PPO update!
            #############################

            if self.enable_epopt:
                eps = (self.epopt_warm_start_eps
                       if epoch < self.epopt_warm_start_epochs else
                       self.epopt_eps)
                self.buf.update_mask(eps)
                self.update_policy(self.buf.mask)
            else:
                self.update_policy()

            self.log_epoch(epoch, start_time)

            self.log_time_stat(epoch, epoch_time, epoch_start_time)

    def learn_sim_distribution(self, sampled_contexts, episode_returns,
                               valid_contexts):
        """Learn simulator distribution.
        2 loops here - for evaluation trajectories, for number of contexts
        we want to see, that is the number of envs we want to sample,
        evaluate the policy, store the total rewards, once done, call
        update distribution function."""
        # TODO: We set eval_eps to 1, should we just remove it?
        for eval_iter in range(self.eval_eps):
            l_dr = []
            all_contexts = []
            valid_ctxs = []

            # This neeeds to be set to False if we are doing
            # discrete distribution style of training
            # With out current set up this doesnt make sense anymore
            # TODO: Remove this?
            # if self.reuse_ctx_samples:
            #    # if we didn't see enough samples during the policy
            #    # rollouts, get more here
            #    n_ctxs = max(
            #        1, self.context_samples - len(sampled_contexts))
            # else:
            n_ctxs = self.context_samples

            for _ in range(n_ctxs):
                env, context = self.context_sampler.sample(
                    return_ctxs=True, distr=self.DR_learn_sample_from)
                # TODO: This should be removed? Its being set inside
                # evaluate_policy
                valid_ctx = True
                if env is None:
                    # assume that the task is not solvable with the
                    # given context, thus it returns the lowest
                    # possible rewards
                    l_dr.append(self.R_min)
                    all_contexts.append(context)
                    valid_ctxs.append(False)
                    continue

                # Otherwise, evaluate the current policy in the given
                # context
                ep_ret, ep_len, valid_ctx, obs = self.evaluate_policy(
                    env, self.get_action_ops)
                all_contexts.append(context)
                valid_ctxs.append(valid_ctx)
                Jpi = self.validate_rollout(ep_ret, ep_len, valid_ctx, obs)

                if self.dr_objective == Objectives.REWARDS:
                    l_dr.append(
                        self.validate_rollout(ep_ret, ep_len, valid_ctx, obs))
                elif self.dr_objective == Objectives.IMPROVEMENT:
                    # evaluate the old policy
                    ep_ret, ep_len, valid_ctx, obs = self.evaluate_policy(
                        env, self.old_action_ops)
                    Jpi_old = self.validate_rollout(
                        ep_ret, ep_len, valid_ctx, obs)
                    rel_improv = (Jpi - Jpi_old)/Jpi
                    l_dr.append(rel_improv)
                else:
                    raise NotImplementedError

            # if self.reuse_ctx_samples:
            #    # append values from policy optimization rollouts
            #    all_contexts += sampled_contexts
            #    total_rewards += episode_returns
            #    valid_ctxs += valid_contexts
            self.update_sim_distribution(all_contexts, l_dr)

    def validate_rollout(self, ep_ret, ep_len, valid_ctx, obs):
        if valid_ctx:
            if ep_len < self.max_ep_len:
                # if trajectory didn't reach terminal state,
                # bootstrap value target
                last_val = self.sess.run(
                    self.v, feed_dict={self.x_ph: obs.reshape(1, -1)})
                ep_ret = ep_ret + np.asscalar(last_val)

            # Also count the rewards during evaluation
            # logger.store(EpRet=ep_ret, EpLen=ep_len)
            if isinstance(ep_ret, np.ndarray):
                ep_ret = np.asscalar(ep_ret)
            return ep_ret
        else:
            return self.R_min

    def evaluate_policy(self, env, action_ops):
        """This function is different from the rollout_policy func,
        as it's called inside distribution learning. For a given
        env, it simply evaluates the policy.
        """
        # The env can fail during step, so we check for
        # `valid_ctx` here as well.
        valid_ctx = True
        context = env.context
        obs, reward, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        if self.cat_context:
            obs = self.concat_context(obs, context)

        if self.normalize_obs:
            self.obs_norm.update(obs)
            obs = self.obs_norm.normalize(obs)

        # This loop will take longer as the policy improves and manages
        # to reach the `self.max_ep_len`
        while not done:
            action, v_t, logp_t = self.sess.run(
                action_ops, feed_dict={self.x_ph: obs.reshape(1, -1)})
            if self.store_context_buffer:
                self.ctx_val_buf.store(
                    obs, action, reward, v_t, logp_t, context)

            try:
                obs, reward, done, _ = env.step(action[0])
                if self.cat_context:
                    obs = self.concat_context(obs, context)
                if self.normalize_obs:
                    obs = self.get_norm_obs(obs)
                ep_ret += reward
                ep_len += 1
            except Exception:
                valid_ctx = False

            if self.store_context_buffer:
                if done or (ep_len == self.max_ep_len):
                    last_val = reward if done else self.sess.run(
                        self.v, feed_dict={self.x_ph: obs.reshape(1, -1)})
                    ep_ret = ep_ret if done else ep_ret + \
                        np.asscalar(last_val)
                    self.ctx_val_buf.finish_path(last_val)

                    # Store context buffer info to file
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    context_buf_filename = self.context_storage_path + \
                        '/' + timestamp + '_val' + '.pkl'
                    with open(context_buf_filename, 'wb') as output:
                        pickle.dump(
                            self.ctx_val_buf,
                            output,
                            pickle.HIGHEST_PROTOCOL)

                    if self.ctx_val_buf.ptr >= self.ctx_val_buf.max_size:
                        # We never call `get` on this buffer so reset manually
                        self.ctx_val_buf.ptr, self.ctx_val_buf.path_start_idx = 0, 0
                        self.ctx_val_buf.path_slices = []

            if (ep_len == self.max_ep_len):
                done = True

        # Also return the latest observations, as it might be needed
        # for bootstrapping value target.
        return ep_ret, ep_len, valid_ctx, obs

    def update_sim_distribution(self, all_contexts, l_dr):
        """Update the running rewards mean, std and minimum reward,
        that is R_min, R_std and R_min respectively. It then evaluates
        the loss and does optimization step for self.distr_learning_iters
        steps.
        """
        p_train = self.context_sampler.train_dist
        p_test = self.context_sampler.test_dist
        R = torch.FloatTensor(l_dr).flatten()

        # update cumulative reward statistics
        beta = 0.01
        self.R_mean = beta * R.mean(0) + (1 - beta) * self.R_mean
        self.R_std = beta * R.std(0) + (1 - beta) * self.R_std
        R_min = np.asscalar((np.atleast_1d(R.min().detach().cpu().numpy())))
        self.R_min = min(self.R_min, R_min)

        if self.whiten_l_dr:
            # compute whitened objective
            R_ = (R - self.R_mean) / (self.R_std + 1e-8)
        else:
            R_ = R

        if self.use_IS_weights and self.DR_learn_sample_from == 'test':
            z = torch.stack(all_contexts).to(dtype=R_.dtype)
            IS_Weights = (p_train.log_prob(z)
                          - p_test.log_prob(z)).exp()
            R_ = R_*IS_Weights

        # overwrite alpha
        alpha = self.alpha / p_test.entropy().abs().detach()

        # Update p(z) using REINFORCE, REBAR or RELAX
        for _ in range(self.distr_learning_iters):
            # zero gradients
            self.distr_optimizer.zero_grad()

            # compute log probs and regularizer
            log_prob = p_train.log_prob(
                torch.stack(all_contexts))

            if torch.any(torch.isnan(log_prob)):
                print('Contexts ', all_contexts)
                raise ValueError('Got Nan log probs')

            entropy = p_train.entropy()

            if self.use_kl_regularizer:
                # empirical kl divergence computation
                z = p_test.sample(torch.Size([1000])).detach()
                log_p_train = p_train.log_prob(z)
                log_p_test = p_test.log_prob(z)
                kl_samples = log_p_test - log_p_train
                kl_loss = kl_samples.mean(0)
                if kl_loss.dim() > 0:
                    # same as before: assuming independence
                    kl_loss = kl_loss.sum(-1)

                regularizer = -kl_loss
            else:
                regularizer = entropy

            if log_prob.dim() > 1:
                # certain distributions, like multidimensional
                # uniform, return different values for each
                # dimension. here, we assume that the log_probs
                # for each dimension are independent of each other,
                # so we can just add them to get the joint log_prob
                log_prob = log_prob.sum(-1)

            if entropy.dim() > 0:
                # same as before: assuming independence, the
                # entropy of the joint distribution equals the sum
                # of the per-dimension entropies
                entropy = entropy.sum(-1)

            loss = -((R_.detach() * log_prob).mean(0) +
                     alpha*regularizer)
            try:
                if self.console_out:
                    print('Rewards ', R)
                    print('Logprob ', log_prob)
                    print('Entropy ', kl_loss)
                loss.backward()
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(R_.shape, log_prob.shape, entropy.shape, loss.shape)
                print('====')
                raise e

            self.distr_optimizer.step()

        if self.console_out:
            print('Loss ', loss)
        self.log_distributions(loss, entropy, p_test.entropy(), l_dr)

    def save_model(self, env):
        def save_or_retry(retries):
            try:
                self.logger.save_state({'env': env}, None)
            except BaseException:
                if retries > 0:
                    print("Saving {0} failed. Retrying...".format(
                        self.logger_kwargs['output_dir']))
                    save_or_retry(retries - 1)
                else:
                    print("Saving {0} failed".format(
                        self.logger_kwargs['output_dir']))

        # the logger fails sometimes if we launch many jobs in parallel
        # (e.g. 12) and all of them try to write on disk with a buggy
        # driver that causes slow writing speeds. Here we try it a
        # couple times, and prevent the code from crashing when there's
        # an IOError
        save_or_retry(3)

        # save context distribution
        torch.save(self.context_sampler,
                   self.logger_kwargs['output_dir'] + "/context_dist.pth")

    def log_epoch(self, epoch, start_time):

        # Log info about epoch
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('VVals', with_min_and_max=True)
        self.logger.log_tabular('TotalEnvInteracts',
                                (epoch + 1) * self.steps_per_epoch)
        self.logger.log_tabular('StopIter', average_only=True)
        self.logger.log_tabular('SeenCtxN', self.ctx_counter)
        self.logger.log_tabular('Time', time.time() - start_time)

        self.logger.dump_tabular(self.console_out)

    def log_time_stat(self, epoch, epoch_time, epoch_start_time):
        if epoch_time is None:
            epoch_time = time.time() - epoch_start_time
        else:
            epoch_time = 0.1 * (
                time.time() - epoch_start_time) + 0.9 * epoch_time
        remaining_time = (epoch_time * (self.epochs - epoch))
        hours, rest = divmod(remaining_time, 3600)
        minutes, sec = divmod(rest, 60)
        # This is the current estimate of whatever process is currently
        # running
        worker_stat = \
            'Training time estimate of worker %d ->' + \
            'hours %d :minutes %d :seconds %d'
        print(worker_stat % (os.getpid(), hours, minutes, sec), end='\r')

    def log_context(self, context, ep_ret):
        context_data = context.data.numpy().flatten()
        for i, z_i in enumerate(context_data):
            col_name = 'Context'
            if i > 0:
                col_name += '_' + str(i)
            self.context_logger.log_tabular(col_name, z_i)
        self.context_logger.log_tabular('Final reward', ep_ret)
        self.context_logger.dump_tabular(console=False)

    def log_distributions(self, loss, entropy, test_dist_entropy,
                          total_rewards):

        self.debugger_logger.log_tabular('R_min', self.R_min)
        self.debugger_logger.log_tabular('R_mean', self.R_mean.data.numpy())
        self.debugger_logger.log_tabular('R_std', self.R_std.data.numpy())
        self.debugger_logger.log_tabular('Loss', loss)
        self.debugger_logger.log_tabular('Test_distr_entr',
                                         test_dist_entropy.data.numpy())
        self.debugger_logger.dump_tabular(console=False)

        reward_variance = np.std(np.asarray(total_rewards))
        if isinstance(self.context_sampler.train_dist,
                      torch.distributions.MultivariateNormal):
            p_train = self.context_sampler.train_dist
            # get the mean and covariance from the distribution
            # parameters
            mu = p_train.loc
            p_train.update_scale_tril()
            L = p_train.scale_tril
            mu_nump = mu.data.numpy()
            cov_diag = torch.diag(
                L.mm(L.t()), 0).data.numpy()

            self.distribution_logger.log_tabular(
                'AvgCtxRewards', round(np.mean(total_rewards), 3))
            self.distribution_logger.log_tabular('Entropy',
                                                 entropy.data.numpy())

            for m in range(len(mu_nump)):
                count = str(m)
                name = 'mu_' + count
                self.distribution_logger.log_tabular(name, mu_nump[m])

            for s in range(len(cov_diag)):
                count = str(s)
                name = 'cov_diag_' + count
                self.distribution_logger.log_tabular(name, cov_diag[s])

        elif isinstance(self.context_sampler.train_dist,
                        torch.distributions.Uniform):
            lo = self.context_sampler.params[0][:, 0].data.numpy()
            hi = self.context_sampler.params[0][:, 1].data.numpy()

            for ii in range(len(lo)):
                count = str(ii)
                name = 'lo_' + count
                self.distribution_logger.log_tabular(name, lo[ii])

            for ii in range(len(hi)):
                count = str(ii)
                name = 'hi_' + count
                self.distribution_logger.log_tabular(name, hi[ii])

        elif isinstance(self.context_sampler.train_dist,
                        torch.distributions.Categorical):
            lo = self.context_sampler.train_dist.lo.data.numpy()
            hi = self.context_sampler.train_dist.hi.data.numpy()
            probs = self.context_sampler.params[0].softmax(-1).data.numpy()

            for ii in range(len(lo)):
                count = str(ii)
                name = 'lo_' + count
                self.distribution_logger.log_tabular(name, lo[ii])

            for ii in range(len(hi)):
                count = str(ii)
                name = 'hi_' + count
                self.distribution_logger.log_tabular(name, hi[ii])

            for ii in range(len(probs)):
                count = str(ii)
                name = 'probs_' + count
                self.distribution_logger.log_tabular(name, probs[ii])

            self.distribution_logger.log_tabular('Entropy',
                                                 entropy.data.numpy())

        self.distribution_logger.log_tabular('CtxRewardStd', reward_variance)
        self.distribution_logger.dump_tabular(self.console_out)

    # Rendering viewer functions

    def init_viewers(self, env, viewers=None):
        if hasattr(env, '_viewers'):
            if viewers is None:
                viewers = env._viewers
            else:
                viewers.update(env._viewers)
        elif hasattr(env, 'viewer'):
            if viewers is None:
                viewers = {}
            viewers['human'] = env.viewer
        return viewers

    def update_viewers(self, env, viewers):
        if hasattr(env, '_viewers'):
            env._viewers = viewers
        elif hasattr(env, 'viewer'):
            env.viewer = viewers['human']
        if isinstance(env, gym.envs.mujoco.MujocoEnv):
            for viewer in viewers.values():
                viewer.update_sim(env.sim)
