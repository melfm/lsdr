import argparse
import gym
import os
import numpy as np
import tensorflow as tf
import time
import torch
import json

from lsdr.algorithm.ppo import core
from lsdr.algorithm.ppo.ppo import PPOBuffer
from lsdr.envs import environment_sampler
from lsdr.envs.environment_sampler import sample_env
from lsdr.utils.logx import restore_tf_graph
from lsdr.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from lsdr.utils.mpi_tools import mpi_avg, num_procs
from lsdr.utils.logx import EpochLogger, Logger
from lsdr.algorithm.ppo.normalizer import Normalizer


def load_trainable_parameters(path, normalize_obs):
    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    # we restore the tf graph, which returns a model. This call also sets the
    # the values of the trainable variables in the graph, which is what we
    # actually need (tf.trainable_variables)
    restore_tf_graph(sess, path)
    variables_to_load = dict(
        [(var.name, sess.run(var)) for var in tf.trainable_variables()])

    if normalize_obs:
        # Load normalizer variables
        norm_mean = tf.global_variables('normalizer/mean')[0]
        variables_to_load.update({norm_mean.name: sess.run(norm_mean)})
        norm_std = tf.global_variables('normalizer/std')[0]
        variables_to_load.update({norm_std.name: sess.run(norm_std)})
    return variables_to_load


def update_trainable_parameter_values(var_dict, graph=None, sess=None):
    if graph is None:
        graph = tf.get_default_graph()

    update_ops = []
    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
        for name in var_dict:
            try:
                tensor = graph.get_tensor_by_name(name)
                update_ops.append(tf.assign(tensor, var_dict[name]))
            except KeyError:
                print('%s does not exist. Not going to be used' % name)

    if sess is None:
        return update_ops
    return sess.run(update_ops)


def init_ops(env,
             logger,
             actor_critic=core.mlp_actor_critic,
             ac_kwargs=dict(),
             steps_per_epoch=4000,
             gamma=0.99,
             clip_ratio=0.2,
             lam=0.97,
             context_dims=0,
             cat_context=False,
             infer_ctx=False):

    obs_dim_orig = list(env.observation_space.shape)
    obs_dim = list(env.observation_space.shape)

    if context_dims > 0:
        obs_dim_orig[0] += context_dims
    obs_dim = tuple(obs_dim)
    act_dim = env.action_space.shape

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    if infer_ctx:
        x_ph, a_ph = core.placeholders_from_spaces(obs_dim,
                                                   env.action_space)
    else:
        x_ph, a_ph = core.placeholders_from_spaces(obs_dim_orig,
                                                   env.action_space)

    adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)

    # TODO: What to do with context inference?
    # Append context to state
    if infer_ctx and context_dims > 0:
        with tf.variable_scope("context"):
            ctx = tf.get_variable(
                'ctx', (context_dims, ), x_ph.dtype, trainable=True)

        # we assume that the placeholders have a batch first dimension
        x = tf.concat(
            [x_ph, tf.tile(ctx[None, :], (tf.shape(x_ph)[0], 1))], -1)

    # Main outputs from computation graph
    # Feed `x` here as it contains both the observations and inferred
    # context.
    if infer_ctx:
        # Feed the augmented input x to the network.
        pi, logp, logp_pi, v = actor_critic(x, a_ph, **ac_kwargs)
    else:
        pi, logp, logp_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    if infer_ctx:
        buf = PPOBuffer(obs_dim, act_dim,
                        local_steps_per_epoch, gamma, lam)
    else:
        buf = PPOBuffer(obs_dim_orig, act_dim,
                        local_steps_per_epoch, gamma, lam)

    # PPO objectives
    # pi(a|s) / pi_old(a|s)
    ratio = tf.exp(logp - logp_old_ph)
    min_adv = tf.where(adv_ph > 0, (1 + clip_ratio) * adv_ph,
                       (1 - clip_ratio) * adv_ph)
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
    v_loss = tf.reduce_mean((ret_ph - v)**2)

    # Info (useful to watch during learning)
    # a sample estimate for KL-divergence, easy to compute
    approx_kl = tf.reduce_mean(logp_old_ph - logp)
    # a sample estimate for entropy, also easy to compute
    approx_ent = tf.reduce_mean(-logp)
    clipped = tf.logical_or(ratio > (1 + clip_ratio), ratio < (1 - clip_ratio))
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

    # Need all placeholders in *this* order later (to zip with data from
    # buffer)
    placeholders = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

    # actor critic ops
    actor_critic_ops = pi, logp, logp_pi, v

    # losses
    losses = [pi_loss, v_loss]

    # statistics
    stats = [approx_kl, approx_ent, clipfrac]

    return placeholders, actor_critic_ops, buf, losses, stats


def build_update_function(all_phs,
                          buf,
                          losses,
                          stats,
                          logger,
                          sess=None,
                          pi_lr=3e-4,
                          vf_lr=1e-3,
                          train_pi_iters=80,
                          train_v_iters=80,
                          target_kl=0.01,
                          learn_ctx=True,
                          fine_tune=True):
    # unpack losses and stats
    pi_loss, v_loss = losses
    approx_kl, approx_ent, clipfrac = stats

    v_trainable_vars = []
    pi_trainable_vars = []
    # One of these needs to be be True, in order to optimize
    if not (learn_ctx or fine_tune):
        raise ValueError('Either fine-tune or learn ctx needs to be enabled.')

    if fine_tune:
        pi_trainable_vars += tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'pi')
        v_trainable_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              'v')
    if learn_ctx:
        v_trainable_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              'context')
        pi_trainable_vars += tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'context')

    train_pi = MpiAdamOptimizer(learning_rate=pi_lr).minimize(
        pi_loss, var_list=pi_trainable_vars)
    train_v = MpiAdamOptimizer(learning_rate=vf_lr).minimize(
        v_loss, var_list=v_trainable_vars)

    # get session
    if sess is None:
        sess = tf.get_default_session()

    def update(all_phs, buf, pi_loss, v_loss, approx_ent, approx_kl, clipfrac,
               train_pi, train_v, sess):
        with sess.as_default():
            inputs = {k: v for k, v in zip(all_phs, buf.get())}
            pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent],
                                              feed_dict=inputs)

            # Train actor
            for i in range(train_pi_iters):
                _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs)
                kl = mpi_avg(kl)
                if kl > 1.5 * target_kl:
                    # logger.log(
                    #    'Early stopping at step %d due to reaching max kl.' %
                    #    i)
                    break
            logger.store(StopIter=i)
            # Train critic
            for _ in range(train_v_iters):
                sess.run(train_v, feed_dict=inputs)

            # Log changes from update
            pi_l_new, v_l_new, kl, cf = sess.run(
                [pi_loss, v_loss, approx_kl, clipfrac], feed_dict=inputs)
            logger.store(
                LossPi=pi_l_old,
                LossV=v_l_old,
                KL=kl,
                Entropy=ent,
                ClipFrac=cf,
                DeltaLossPi=(pi_l_new - pi_l_old),
                DeltaLossV=(v_l_new - v_l_old))

    from functools import partial
    # bind variables to update function handle
    update_fn = partial(update, all_phs, buf, pi_loss, v_loss, approx_ent,
                        approx_kl, clipfrac, train_pi, train_v, sess)

    return update_fn


def ppo(env_fn,
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
        save_freq=10,
        render=False,
        context_dims=0,
        pretrained_path=None,
        console_out=True,
        infer_ctx=False,
        learn_ctx=True,
        fine_tune=True,
        normalize_ctx=True,
        normalize_obs=False,
        n_eval_trajs=1):

    tf.reset_default_graph()
    logger_kwargs.update({'output_fname': 'transfer.txt'})
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    if infer_ctx and normalize_obs:
        # Currently the observation normalizer requires the full state
        # including the context. This doesn't work with the way context
        # inference is setup. This can be fixed in the future, but since
        # the implicit context inference was inaccurate, here we just
        # append the actual contexts.
        raise ValueError('Do not enable obs normalization with ctx learning.')

    if learn_ctx:
        ctx_logger = Logger(
            output_dir=logger_kwargs['output_dir'],
            output_fname='inferred_ctxs.txt')

    env = env_fn()

    # Find the source exp config file
    output_dir = logger_kwargs['output_dir']
    src_path = os.path.split(os.path.split(os.path.split(output_dir)[0])[0])[0]
    src_name = os.path.split(src_path)[1]
    # TODO: We also need the trainer seed - for now assume always 0
    src_path = src_path + '/training/' + src_name + '_s0'
    json_file = src_path + '/' + 'config.json'
    with open(json_file) as f:
        data = json.load(f)

    hidden_sizes = data['ac_kwargs']['hidden_sizes']
    ac_kwargs['hidden_sizes'] = hidden_sizes

    # init graph inputs and outputs
    target_graph = tf.get_default_graph()
    all_phs, actor_critic_ops, buf, losses, stats = init_ops(
        env, logger, actor_critic, ac_kwargs, steps_per_epoch, gamma,
        clip_ratio, lam, context_dims, infer_ctx=infer_ctx)
    x_ph, a_ph, adv_ph, ret_ph, logp_old_ph = all_phs
    pi, logp, logp_pi, v = actor_critic_ops
    local_steps_per_epoch = buf.max_size

    if infer_ctx and context_dims > 0:
        with tf.variable_scope('context', reuse=True):
            ctx = tf.get_local_variable("ctx")

    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi]
    # initialize session
    sess = tf.Session()
    if normalize_obs:
        with tf.variable_scope("normalizer"):
            obs_dim = env.observation_space.shape[0]
            if context_dims > 0:
                obs_dim += context_dims
            obs_norm = Normalizer(obs_dim, sess=sess)

    init_params_ops = []
    if pretrained_path is not None:
        if console_out:
            print("Loading parameters from %s " % (pretrained_path))

        # load graph from file into source graph
        source_graph = tf.Graph()
        with source_graph.as_default():
            params_dict = load_trainable_parameters(pretrained_path,
                                                    normalize_obs)

        # load context distribution
        if context_dims > 0:
            try:
                base_path = os.path.dirname(pretrained_path)
                # The context distribution is located at a dir
                # one level above so skip dir 'simple_save'
                base_path = base_path.split('simple_save')[0]
                context_sampler = torch.load(
                    os.path.join(base_path, 'context_dist.pth'))

                context_min = context_sampler.test_dist.lo
                context_max = context_sampler.test_dist.hi
                context_range = context_max - context_min
                test_ctx = env.context

                if normalize_ctx:
                    test_ctx = (test_ctx - context_min) / context_range
                if not normalize_obs:
                    params_dict[ctx.name] = test_ctx.cpu().detach().numpy()
            except FileNotFoundError:
                # If the distribution file doesn't exist, just append
                # the true context from `env`, by uncommenting the following
                # two lines. This would also mean that we can't normalize
                # the context.
                # test_ctx = env.context
                # params_dict[ctx.name] = test_ctx.cpu().detach().numpy()
                raise ValueError('Context distribution file not found')

        # load pre trained params into target graph (will need to run the
        # init_params_ops in a session)
        init_params_ops = update_trainable_parameter_values(
            params_dict, target_graph)

    # define ppo update funcion
    update = build_update_function(
        all_phs,
        buf,
        losses,
        stats,
        logger,
        sess,
        pi_lr,
        vf_lr,
        train_pi_iters,
        train_v_iters,
        target_kl,
        learn_ctx=learn_ctx,
        fine_tune=fine_tune)

    # initialize variables
    sess.run(tf.global_variables_initializer())

    # load pre-trained parameters
    if len(init_params_ops) > 0:
        sess.run(init_params_ops)

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v})

    epoch_time = None
    start_time = time.time()

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        epoch_start_time = time.time()
        obs, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        # If we are inferring context, it should already be part of
        # observation in the case of context-conditioned training
        if not infer_ctx:
            if context_dims > 0:
                obs = np.append(obs, test_ctx)
            if normalize_obs:
                # Note: The normalizer needs the full state space.
                obs = obs_norm.normalize(obs)
        n_trajs = 0
        for t in range(local_steps_per_epoch):
            # evaluate policy and value
            a, v_t, logp_t = sess.run(
                get_action_ops, feed_dict={x_ph: obs.reshape(1, -1)})
            # append data to experience buffer
            buf.store(obs, a, r, v_t, logp_t)
            # apply action
            obs, r, d, _ = env.step(a[0])
            if not infer_ctx:
                if context_dims > 0:
                    obs = np.append(obs, test_ctx)
                if normalize_obs:
                    obs = obs_norm.normalize(obs)
            # log value
            ep_ret += r
            ep_len += 1
            logger.store(VVals=v_t)
            if render:
                env.render()

            # check if reached terminal state
            terminal = d or (ep_len == max_ep_len)
            if terminal or (t == local_steps_per_epoch - 1):
                if not (terminal) and console_out:
                    print('Warning: trajectory cut off by epoch at \
                            %d steps.' % ep_len)
                # if trajectory didn't reach terminal state, bootstrap
                # value target
                last_val = r if d else sess.run(
                    v, feed_dict={x_ph: obs.reshape(1, -1)})
                buf.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                # reset environment
                n_trajs += 1
                obs, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                if not infer_ctx:
                    if context_dims > 0:
                        obs = np.append(obs, test_ctx)
                    if normalize_obs:
                        obs = obs_norm.normalize(obs)

        # Add more evaluation trajectories if needed
        # More trajectories == better estimate of AverageEpRet
        # If the policy is doing a good job, we get trajectories of length
        # 'max_ep_len'. In contrast, if its doing a bad job and each rollout
        # gets cut short early we get a lot more trajectories.
        # The estimate of the mean return has less variance if we have a
        # lot of samples, so bump up the number of trajectories.
        while n_trajs < n_eval_trajs:
            obs, reward, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            if context_dims > 0:
                obs = np.append(obs, test_ctx)
            obs = obs_norm.normalize(obs)
            while not done:
                # evaluate policy
                a = sess.run(pi, feed_dict={x_ph: obs.reshape(1, -1)})
                obs, reward, done, _ = env.step(a[0])
                if not infer_ctx:
                    if context_dims > 0:
                        obs = np.append(obs, test_ctx)
                    if normalize_obs:
                        obs = obs_norm.normalize(obs)
                ep_ret += reward
                ep_len += 1
                if ep_len == max_ep_len:
                    done = True
            n_trajs += 1
            logger.store(EpRet=ep_ret, EpLen=ep_len)

        # Save model and env state
        if (epoch % save_freq == 0) or (epoch == epochs - 1):

            def save_or_retry(retries):
                try:
                    logger.save_state({'env': env}, None)
                except BaseException:
                    if retries > 0:
                        print("Saving {0} failed. Retrying...".format(
                            logger_kwargs['output_dir']))
                        save_or_retry(retries - 1)
                    else:
                        print("Saving {0} failed".format(
                            logger_kwargs['output_dir']))

            save_or_retry(3)

        # Perform PPO update!
        update()

        if learn_ctx:
            est_ctx = sess.run(ctx)
            if console_out:
                print("Context after update %s" % (est_ctx))

            for c in range(len(est_ctx)):
                ctx_name = 'Inferred_context_' + str(c)
                ctx_logger.log_tabular(ctx_name, est_ctx[c])
            ctx_logger.dump_tabular(console=False)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular(console=False)

        if epoch_time is None:
            epoch_time = time.time() - epoch_start_time
        else:
            epoch_time = 0.1 * (
                time.time() - epoch_start_time) + 0.9 * epoch_time

        remaining_time = (epoch_time * (epochs - epoch))
        hours, rest = divmod(remaining_time, 3600)
        minutes, sec = divmod(rest, 60)
        print(
            'Transfer time of worker %d -> hours %d :minutes %d :seconds %d' %
            (os.getpid(), hours, minutes, sec),
            end='\r')


if __name__ == '__main__':
    # Example usage:
    # python ppo_transfer.py  --pretrained_path data/path/simple_save
    # --exp_name my_transfer_exp_name --fine_tune
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='cartpole')
    parser.add_argument('--l', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='ppo_transfer')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--learn_ctx', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    from lsdr.utils.run_utils import setup_logger_kwargs

    # Find the exp source name from the pretrained path
    exp_src = args.pretrained_path.split('/')[1]

    logger_kwargs = setup_logger_kwargs(
        exp_name=args.exp_name,
        phase='transfer',
        exp_src=exp_src,
        seed=args.seed)

    if args.env in environment_sampler.available_envs:

        context_sampler = environment_sampler.init_env_sampler(
            args.env, args.seed, args.exp_name)
        env = sample_env(context_sampler, distr='train')

        context_dims = len(env.context)
    else:
        env = gym.make(args.env)
        context_dims = args.context_dims

    ppo(lambda: env,
        actor_critic=core.mlp_actor_critic,
        gamma=args.gamma,
        seed=args.seed,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
        context_dims=context_dims,
        pretrained_path=args.pretrained_path,
        render=args.render,
        learn_ctx=args.learn_ctx,
        fine_tune=args.fine_tune)
