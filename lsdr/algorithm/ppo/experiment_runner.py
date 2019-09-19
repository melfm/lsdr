import numpy as np
import gym
import json
import os
import multiprocessing
import tensorflow as tf
import time
from collections import OrderedDict

import lsdr.algorithm.ppo.ppo as ppo
import lsdr.algorithm.ppo.ppo_transfer as ppo_transfer

import lsdr.algorithm.ppo.core as core
from lsdr.envs import environment_sampler
from lsdr.envs.environment_sampler import init_env_sampler
from lsdr.utils.logx import colorize
from lsdr.utils.run_utils import setup_logger_kwargs

Debug_P = False
console_log = True
file_wr_lock = None


def init_pool_vars(lock):
    global file_wr_lock
    file_wr_lock = lock


def ppo_call(args,
             experiment_id,
             exp_params,
             dist_params=None,
             test_dist_params=None,
             dist_type=None,
             dist_param_idxs=None,
             skip_train=False,
             store_context_buffer=False):

    default_env = False
    learn_distr = False

    if skip_train:
        print(colorize("skipping training ...", 'yellow', bold=True))
        return

    if console_log:
        print(
            colorize(
                "PPO Training: Launching {0}.".format(experiment_id),
                'cyan',
                bold=True))

    try:
        logger_kwargs = setup_logger_kwargs(
            exp_name=experiment_id,
            phase='training',
            seed=args.seed,
            data_dir=args.data_dir)

        if exp_params['train_ph'] == 'NONE':
            learn_distr = False
            default_env = True
        elif exp_params['train_ph'] == 'DR_FIXED':
            learn_distr = False
        elif exp_params['train_ph'] == 'DR_LEARNED':
            learn_distr = True

        context_sampler = init_env_sampler(
            exp_params['env'],
            args.seed,
            experiment_id,
            default_env=default_env,
            init_dist_params=dist_params,
            test_dist_params=test_dist_params,
            dist_type=dist_type)

        hidden_sizes = list(map(int, args.hid))
        ppo_trainer = ppo.PPO(
            context_sampler,
            actor_critic=core.mlp_actor_critic,
            ac_kwargs=dict(hidden_sizes=hidden_sizes),
            logger_kwargs=logger_kwargs,
            learn_distr=learn_distr,
            dist_param_idxs=dist_param_idxs,
            console_out=False,
            store_context_buffer=store_context_buffer,
            **vars(args))

        ppo_trainer.train_ppo()

        print(
            colorize(
                "PPO Training: Finished {0} Seed {1}.".format(
                    experiment_id, args.seed),
                'green',
                bold=True))
    except Exception as e:
        print(
            colorize(
                'PPO Training: Something went wrong :( {0} {1}'.format(
                    experiment_id, args.seed),
                'red',
                bold=True))
        import traceback
        traceback.print_exc()
        raise e

    return time.time()


def ppo_transfer_call(args,
                      experiment_id,
                      exp_params,
                      tran_dist_params=None,
                      dist_type=None,
                      train_experiment_id=None,
                      exp_seed=None,
                      cat_context=False):

    global file_wr_lock
    experiment_id_w_seed = experiment_id + '_s' + str(exp_seed) \
        if exp_seed is not None else experiment_id

    if console_log:
        print(
            colorize(
                "PPO:Tran: Launching {0}.".format(experiment_id_w_seed),
                'green',
                bold=True))

    try:
        # start test
        if exp_seed is not None:
            args.seed = exp_seed

        # building path for loading source training data
        training_src_path = setup_logger_kwargs(
            exp_name=train_experiment_id,
            phase='training',
            seed=0,
            data_dir=args.data_dir)['output_dir'] + '/simple_save'

        logger_kwargs = setup_logger_kwargs(
            exp_name=experiment_id,
            phase='transfer',
            exp_src=train_experiment_id,
            seed=args.seed,
            data_dir=args.data_dir)

        if os.path.isdir(logger_kwargs['output_dir']):
            # load data, check if length of data == test epochs
            try:
                import pandas as pd
                data = pd.read_table(
                    os.path.join(logger_kwargs['output_dir'], 'transfer.txt'))

                if len(data) >= args.test_epochs:
                    print(
                        colorize(
                            "PPO:Tran: skipping {0}. Data exists.".format(
                                experiment_id_w_seed),
                            'yellow',
                            bold=True))
                    return time.time()
            except Exception:
                pass

        if exp_params['env'] in environment_sampler.available_envs:
            # During transfer, overwrite the test dist config with a solvable
            # range. To fix this, we can add an additional final test distr.
            if dist_type == 'discrete':
                # Overwrite to uniform to make sure same test envs are
                # sampled from a uniform distribution to make it comparable
                # against the domain randomization.
                dist_type = 'uniform'
            context_sampler = init_env_sampler(
                exp_params['env'],
                args.seed,
                experiment_id_w_seed,
                default_env=None,
                init_dist_params=tran_dist_params,
                test_dist_params=tran_dist_params,
                dist_type=dist_type)

            env = environment_sampler.sample_env(context_sampler,
                                                 distr='test')
            if cat_context:
                context_dims = len(env.context)
            else:
                context_dims = 0

        else:
            env = gym.make(exp_params['env'])
            context_dims = args.context_dims

        sampled_envs_json = os.path.join(
            os.path.dirname(logger_kwargs['output_dir']), 'sampled_envs.json')
        if file_wr_lock is not None:
            file_wr_lock.acquire()
        try:
            os.makedirs(os.path.dirname(logger_kwargs['output_dir']))
        except BaseException:
            pass
        try:
            with open(sampled_envs_json, 'r') as f:
                sampled_envs = json.load(f)
        except BaseException:
            sampled_envs = {}

        sampled_envs[args.seed] = env.context.tolist()

        with open(sampled_envs_json, 'w+') as f:
            json.dump(sampled_envs, f)

        if file_wr_lock is not None:
            file_wr_lock.release()
        if not(cat_context) and args.learn_ctx:
            raise ValueError(
                'Cannot run no context experiment with learn ctx enabled!')

        norm_obs = args.normalize_obs
        ppo_transfer.ppo(
            lambda: env,
            actor_critic=core.mlp_actor_critic,
            epochs=args.test_epochs,
            logger_kwargs=logger_kwargs,
            context_dims=context_dims,
            pretrained_path=training_src_path,
            normalize_ctx=args.normalize_ctx,
            learn_ctx=args.learn_ctx,
            console_out=False,
            normalize_obs=norm_obs)

        print(
            colorize(
                "PPO:Tran: Finished {0}.".format(experiment_id_w_seed),
                'green',
                bold=True))
    except Exception as e:
        print(
            colorize(
                'PPO:Tran: Something went wrong :( {0}'.format(
                    experiment_id_w_seed),
                'red',
                bold=True))
        import traceback
        traceback.print_exc()
        raise e

    return time.time()


def callbackfunct(x):
    if x:
        print(x)


def launch_train_exp_runs(dist_dict,
                          test_dist_dict,
                          cat_context,
                          train_family,
                          exp_params,
                          exp_args,
                          unique_exp_id,
                          exp_key,
                          exp_runs,
                          pool,
                          res,
                          normalize_ctx=False,
                          normalize_obs=False,
                          all_runs=True,
                          run_ids=None,
                          skip_train=False,
                          store_ctx_buffer=False,
                          distr_learning_iters=10):
    """
    Args:
        dist_dict: (dict), indexed by run id for key, and the distribution
            to sample from.
        cat_context: (Boolean), flag to run experiment with context appended
            to the observations.
        train_family: (String), train distribution to sample from, 'gaussian'
            or 'uniform'.
        exp_params: (dict), containing experiment parameters.
        exp_args: (argparse), containing experiment setup read from existing
            json files - e.g. train_ctx_configs.json
        unique_exp_id: (str), unique id of experiment.
        exp_key: (str), key identifying the experiment id.
        exp_runs: (int), number of experiment runs. This is the number of times
            each experiment is repeated, determined from the experiment
            distribution json file.
        pool: (Multiprocessing pool), an instance of Pool object to keep track
            of worker processes.
        res: (dict), results from multiprocessing.apply_async().
        all_runs: (Boolean), flag to start all the runs
        run_ids: (list of int), run ids to launch (overrridden by the all_runs argument).
    """

    def run_experiment(dist_params, test_dist_params, param_idxs, run):
        # ID: Exp:xxx_family_phase_run_x
        experiment_id = unique_exp_id + '_' + exp_key.replace(":", "_") \
            + '_' + train_family + '_'
        experiment_id += exp_params['train_ph']

        if cat_context:
            experiment_id += '_context'

        experiment_id += '_run_' + str(run)

        if Debug_P:
            ppo_call(
                exp_args,
                experiment_id,
                exp_params,
                dist_params,
                test_dist_params,
                train_family,
                dist_param_idxs=param_idxs,
                skip_train=skip_train,
                store_context_buffer=store_ctx_buffer)
            exp_runs[exp_key + str(run)] = experiment_id
        else:
            # print("Launched experiment {0}".format(experiment_id))
            res[experiment_id] = pool.apply_async(
                ppo_call,
                args=(exp_args, experiment_id, exp_params, dist_params,
                      test_dist_params, train_family,
                      param_idxs, skip_train, store_ctx_buffer),
                callback=callbackfunct)
            exp_runs[exp_key + str(run)] = experiment_id

    if all_runs:
        # launch experiments for all different runs
        for run, dist_config in dist_dict.items():
            # TODO: Make this as option to occur for discrete training
            # only
            for test_run, test_dist_config in test_dist_dict.items():
                if run == test_run:
                    test_dist_params = \
                        deserialize_arrays(test_dist_config['params'])

            train_dist_params = deserialize_arrays(dist_config['params'])
            param_idxs = dist_config.get('trainable_indices', None)
            run_experiment(train_dist_params, test_dist_params, param_idxs,
                           run)

    else:
        if run_ids is None:
            raise ValueError(
                'You must specify the run ids if all_runs is False.')
        # Just run the specified run_ids
        for single_run in run_ids:
            # Look-up the run id, run single experiment
            test_dist_config = test_dist_dict.get(single_run)
            dist_config = dist_dict.get(single_run)
            train_dist_params = deserialize_arrays(dist_config['params'])
            test_dist_params = None
            if test_dist_config is not None:
                test_dist_params = deserialize_arrays(
                    test_dist_config['params'])
            param_idxs = dist_config.get('trainable_indices', None)
            run_experiment(train_dist_params, test_dist_params, param_idxs,
                           single_run)


def launch_test_exp_runs(dist_dict,
                         cat_context,
                         test_family,
                         exp_runs,
                         unique_exp_id,
                         exp_key,
                         exp_params,
                         train_exp_key,
                         pool,
                         res,
                         start_times,
                         test_seeds,
                         all_runs=True,
                         run_ids=None,
                         train_only=False,
                         skip_train=False):
    """
    Args:
        dist_dict: (dict), indexed by run id for key, and the distribution
            to sample from.
        cat_context: (Boolean), flag to run experiment without context appended
            to the observations.
        test_family: (String), train distribution to sample from, 'gaussian'
            or 'uniform'.
        exp_params: (dict), containing experiment parameters.
        exp_args: (argparse), containing experiment setup read from existing
            json files - e.g. train_ctx_configs.json
        unique_exp_id: (str), unique id of experiment.
        exp_key: (str), key identifying the experiment id.
        exp_runs: (int), number of experiment runs. This is the number of times
            each experiment is repeated, determined from the experiment
            distribution json file.
        pool: (Multiprocessing pool), an instance of Pool object to keep track
            of worker processes.
        res: (dict), results from multiprocessing.apply_async().
        test_seeds: (int), different seeds to create the environments during
            test phase.
        all_runs: (Boolean), flag to start all the runs
        run_ids: (list of int), run ids to launch (overrridden by the all_runs argument).
    """

    if train_only:
        print(colorize("Wont transfer ...", 'yellow', bold=True))
        return

    def run_test_experiment(dist_params, run, train_run_id=None):

        # ID: Exp:json-name_family_phase_run_x
        if cat_context:
            experiment_id = unique_exp_id + '_' + exp_key + '_' + \
                test_family + '_' + exp_params['test_ph'] + \
                '_context_run_' + str(run)
        else:
            experiment_id = unique_exp_id + '_' + exp_key + '_' + \
                test_family + '_' + exp_params['test_ph'] + '_run_' + \
                str(run)

        if train_run_id is None:
            key = train_exp_key + str(run)
        else:
            key = train_exp_key + str(train_run_id)
        train_experiment_id = exp_runs[key]

        exp_args.learn_ctx = exp_params['test_ph'].find('LEARN_CTX') > -1
        exp_args.fine_tune = exp_params['test_ph'].find('FINE_TUNE') > -1

        if not Debug_P:
            if not skip_train:
                if console_log:
                    print(
                        colorize(
                            "Transfer waiting until {0} is done.".format(
                                train_experiment_id),
                            'gray',
                            bold=True))
                # wait until the training experiment is done
                res[train_experiment_id].wait()

        for seed in range(0, test_seeds):
            exp_seed = seed
            if Debug_P:
                ppo_transfer_call(
                    exp_args,
                    experiment_id,
                    exp_params,
                    dist_params,
                    test_family,
                    train_experiment_id,
                    exp_seed=exp_seed,
                    cat_context=cat_context)
            else:
                # print("Launched experiment {0}".format(exp_id))
                experiment_id_w_seed = experiment_id + '_' + str(exp_seed)
                start_times[experiment_id_w_seed] = time.time()
                res[experiment_id_w_seed] = pool.apply_async(
                    ppo_transfer_call,
                    args=(exp_args, experiment_id, exp_params, dist_params,
                          test_family, train_experiment_id, exp_seed,
                          cat_context),
                    callback=callbackfunct)

    if all_runs:
        for run, dist_config in dist_dict.items():
            test_dist_params = deserialize_arrays(dist_config['params'])
            run_test_experiment(test_dist_params, run,
                                dist_config.get('train_run_id', None))

    else:
        if run_ids is None:
            raise ValueError(
                'You must specify the run ids if all_runs is False.')
        # Just run the specified run_ids
        for single_run in run_ids:
            # Look-up the run id, run single experiment
            if single_run in dist_dict:
                dist_config = dist_dict.get(single_run)
                test_dist_params = deserialize_arrays(dist_config['params'])
                run_test_experiment(test_dist_params, single_run,
                                    dist_config.get('train_run_id', None))


def run_experiments(script_args, exp_args):

    exp_json_path = script_args.experiments_config
    run_ids = script_args.run
    all_runs = script_args.all_runs
    test_seeds = script_args.test_seeds
    train_only = script_args.train_only
    skip_train = script_args.skip_train
    store_ctx_buffer = script_args.store_ctx_buffer

    print(
        colorize(
            "Experiment %s in progress ..." % exp_json_path,
            'magenta',
            bold=True))
    unique_exp_id = exp_json_path.split('/')[1].split('.')[0]
    cat_context = exp_args.cat_context

    cpu_count = script_args.cpu_cores if script_args.cpu_cores is not \
        None else multiprocessing.cpu_count()
    print("Initializing multiprocessing pool with {0} slots".format(cpu_count))
    file_wr_lock = multiprocessing.Lock()
    pool = multiprocessing.Pool(
        cpu_count, initializer=init_pool_vars, initargs=(file_wr_lock, ))

    with open(exp_json_path, 'r') as f:
        # TODO: Adding extra tran distributions, breaking old exp configs
        train_experiments, test_experiments, train_distributions, \
            test_distributions, tran_distributions = json.load(f)

    res = {}
    exp_runs = {}

    start_times = {}

    for exp_key in train_experiments:
        exp_params = train_experiments[exp_key]
        train_family = exp_params['train_family']

        for test_exp_key in test_experiments:
            test_exp_params = test_experiments[test_exp_key]
            train_exp_key = test_exp_params['train_experiment']

            if train_exp_key == exp_key:
                test_family = test_exp_params['test_family']
                test_dist_dict = OrderedDict()
                for d in test_distributions:
                    dist_config = test_distributions[d]
                    if dist_config['type'] == test_family:
                        run_id = dist_config['run']
                        test_dist_dict[run_id] = dist_config

        # get training distributions
        dist_dict = OrderedDict()
        for d in train_distributions:
            dist_config = train_distributions[d]
            if train_distributions[d]['type'] == train_family:
                run_id = dist_config['run']
                dist_dict[run_id] = dist_config

        launch_train_exp_runs(
            dist_dict,
            test_dist_dict,
            cat_context,
            train_family,
            exp_params,
            exp_args,
            unique_exp_id,
            exp_key,
            exp_runs,
            pool,
            res,
            all_runs=all_runs,
            run_ids=run_ids,
            skip_train=skip_train,
            store_ctx_buffer=store_ctx_buffer)

    for exp_key in test_experiments:
        exp_params = test_experiments[exp_key]
        test_family = exp_params['test_family']
        train_exp_key = exp_params['train_experiment']

        # Sample from distribution
        dist_dict = OrderedDict()
        for d in tran_distributions:
            dist_config = tran_distributions[d]
            if dist_config['type'] == test_family:
                run_id = dist_config['run']
                dist_dict[run_id] = dist_config

        launch_test_exp_runs(
            dist_dict,
            cat_context,
            test_family,
            exp_runs,
            unique_exp_id,
            exp_key,
            exp_params,
            train_exp_key,
            pool,
            res,
            start_times,
            test_seeds,
            all_runs=all_runs,
            run_ids=run_ids,
            train_only=train_only,
            skip_train=skip_train)

    end_times = {}

    for i, (experiment_id, r) in enumerate(res.items()):
        end_times[experiment_id] = r.get()
        if console_log:
            print(
                colorize(
                    "Experiment {0} of {1}.".format(i + 1, len(res)),
                    'white',
                    bold=True))

    pool.close()
    pool.join()
    print(colorize('Done.', 'blue'))

    exp_file_id = exp_json_path.split('/')[1].split('.')[0]
    file_dir = 'time_logs/'
    filename = file_dir + exp_file_id + '_runtime_log.txt'
    os.makedirs(file_dir, exist_ok=True)
    # Create a clean time logger
    if os.path.exists(filename):
        os.remove(filename)

    for exp_id, exp_run_t in start_times.items():
        # Get the experiment run time
        exp_end_t = end_times[exp_id]
        diff_t = exp_end_t - exp_run_t

        with open(filename, "a") as f:
            f.write(exp_id + ': ' + str(diff_t) + '\n')


def deserialize_arrays(string):
    return [
        np.array(data, dtype=dtype).reshape(shape)
        for (dtype, shape, data) in json.loads(string)
    ]


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--experiments_config', type=str, default='configs/experiments.json')
    parser.add_argument(
        '--ctx_config', type=str, default='configs/train_settings/train_ctx_configs.json')

    parser.add_argument('--render', action='store_true')
    parser.add_argument('--distr_learning_iters', type=int, default=10)
    parser.add_argument('--test_epochs', type=int, default=50)
    parser.add_argument('--test_seeds', type=int, default=50)
    parser.add_argument('--run', type=int, default=[0], nargs='+')
    parser.add_argument('--all_runs', action='store_true')
    parser.add_argument('--train_only', action='store_true')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--store_ctx_buffer', action='store_true')
    # May want to customize this for running experiments on the server
    parser.add_argument('--cpu_cores', type=int)

    script_args = parser.parse_args()
    network_config = script_args.ctx_config

    unknown = []
    with open(network_config, 'r') as f:
        config_json = json.load(f)

        parser.set_defaults(**config_json)

        [
            parser.add_argument(arg)
            for arg in [arg for arg in unknown if arg.startswith('--')]
            if arg.split('--')[-1] in config_json
        ]

        exp_args = parser.parse_args()

    if script_args.cpu:
        # Disable GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print('Not using GPU!')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    run_experiments(script_args, exp_args)
