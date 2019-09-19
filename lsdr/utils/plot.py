import matplotlib
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np
from ast import literal_eval
# import torch
import re
from collections import OrderedDict

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()


def get_datasets(logdir, condition=None, ignore_distr=False):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger.

    Assumes that any file "progress.txt" is a valid hit.
    """
    global exp_idx
    global units
    datasets = []

    exp_name = None
    for root, xx, files in os.walk(logdir):

        if 'progress.txt' in files:

            try:
                config_path = open(os.path.join(root, 'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
            except BaseException:
                print('No file named config.json')
            condition1 = condition or exp_name or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            exp_data = pd.read_table(os.path.join(root, 'progress.txt'))

            performance = 'AverageTestEpRet' if 'AverageTestEpRet' in \
                exp_data else 'AverageEpRet'
            exp_data.rename(
                inplace=True, columns={
                    'TotalEnvInteracts': 'Iteration'})
            exp_data.insert(len(exp_data.columns), 'Unit', unit)
            exp_data.insert(len(exp_data.columns), 'Condition1', condition1)
            exp_data.insert(len(exp_data.columns), 'Condition2', condition2)
            exp_data.insert(
                len(exp_data.columns),
                'Average Cumulative Reward', exp_data[performance])
            datasets.append(exp_data)

            if 'distributions.txt' in files and not ignore_distr:
                exp_data_distr = pd.read_table(
                    os.path.join(root, 'distributions.txt'))
                return datasets, exp_data_distr

    if ignore_distr:
        return datasets
    else:
        return datasets, []


def get_all_datasets(all_logdirs,
                     legend=None,
                     select=None,
                     exclude=None,
                     ignore_distr=False):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is,
           pull data from it;

        2) if not, check to see if the entry is a prefix for a
           real directory, and pull data from that.
    """

    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1] == '/':
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)

            def fulldir(x):
                return osp.join(basedir, x)

            prefix = logdir.split('/')[-1]
            listdir = os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])
    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [
            log for log in logdirs if all(not (x in log) for x in exclude)
        ]

    # Verify logdirs
    print('Plotting from...\n' + '=' * DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '=' * DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not(legend) or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = []
    if legend:

        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg, ignore_distr=ignore_distr)
    else:

        for log in logdirs:
            data += get_datasets(log, ignore_distr=ignore_distr)
    return data


def plot_context_distributions(context_data, dirname):

    if "lo_0" in context_data.columns and "probs_0" in context_data.columns:
        # plotting a discrete distribution
        all_lo = sorted([col for col in context_data if col.startswith('lo')])
        all_hi = sorted([col for col in context_data if col.startswith('hi')])
        ndims = len(all_lo)
        dims = list(range(1, ndims+1))

        # convert flattened probabilities array in to multidimensional
        distr_data = np.array(context_data.iloc[:, 2*ndims:-2])
        cells_per_dim = int(np.round(distr_data.shape[1]**(1/ndims)))
        # distr data will be of shape [opt_iters, cells_per_dim, cells_per_dim,
        # ..., cells_per_dim]
        distr_data = distr_data.reshape(-1, *([cells_per_dim]*ndims))

        for i, (lo, hi) in enumerate(zip(all_lo, all_hi)):
            plt.figure()

            lo_, hi_ = context_data[lo][0], context_data[hi][0]
            # marginal distribution for dimension i equals to taking the
            # joint distribution (distr_data) and adding the values over all
            # dimensions except dimension i
            marginal_dist = np.einsum(distr_data, [..., *dims], [..., i+1])

            # renormalizing to deal with numerical accumulation errors
            # marginal_dist /= marginal_dist.sum(-1, keepdims=True)
            plt.imshow(marginal_dist.transpose(), aspect='auto')
            plt.colorbar()

            # Changing the tick to match the context ranges
            # The division by 10 is for readibility purposes!
            step = (hi_ - lo_)/10
            bin_intervals = np.arange(lo_, hi_ + step, step)
            labels = ['%.2e' % label for label in bin_intervals]
            plt.yticks(range(0, cells_per_dim+1, int(cells_per_dim/10)),
                       labels)

            context_code = 'Z' + str(i+1)
            context_title = context_code + ' distribution'
            plt.title(context_title)

            plt.xlabel('Timestep')
            plt.ylabel('Probabilities')
            plotname = dirname + '/' + \
                'Context_' + context_code + '.png'
            plt.savefig(plotname)

            ####################################
            # Calculate final mean and variance
            ####################################
            total_mean = 0
            total_variance = 0
            # Re-calculate the actual step and intervals
            step = (hi_ - lo_)/cells_per_dim
            bin_intervals = np.arange(lo_, hi_ + step, step)
            total_mean = 0
            total_variance = 0
            # Final distribution
            marginal_prob_last = marginal_dist[-1, :]
            max_bin_range = len(marginal_prob_last)
            if len(bin_intervals) % 2 != 0:
                max_bin_range = len(bin_intervals) - 1
            for i in range(max_bin_range - 1):
                min_range = bin_intervals[i]
                max_range = min_range + step
                bin_mean = (min_range + max_range)/2
                total_mean += marginal_prob_last[i] * bin_mean
                total_variance += (marginal_prob_last[i] * bin_mean**2)

            print('Last probabilities', marginal_prob_last)
            print('Context distr mean', total_mean)
            prob_weight_sum = np.sum(marginal_prob_last)
            print('Sum of weights', prob_weight_sum)
            total_variance = (total_variance - total_mean**2)/prob_weight_sum
            print('Context dists variance', total_variance)
            max_range = (2*total_mean + (12*total_variance)**0.5)/2
            min_range = 2*total_mean - max_range
            print('Range -> [', min_range, ' ', max_range, ']')

    else:
        all_means = [col for col in context_data if col.startswith('mu')]

        timesteps = list(range(0, len(context_data)))

        for mu in range(len(all_means)):

            mu_name = 'mu_' + str(mu)
            cov_name = 'cov_diag_' + str(mu)
            context_mu = context_data[mu_name]
            context_cov = np.sqrt(context_data[cov_name])

            plt.figure()
            l, = plt.plot(timesteps, context_mu, linestyle='-')
            for i in range(1, 3):
                plt.fill_between(
                    timesteps,
                    context_mu + i * context_cov,
                    context_mu - i * context_cov,
                    linestyle='None',
                    alpha=0.4**i,
                    color='skyblue')

            context_code = 'Z' + str(mu + 1)
            context_title = context_code
            plt.title(context_title)
            plt.xlabel('Timestep', fontsize=14)
            plt.ylabel('Mean/Variance', fontsize=14)
            plt.tight_layout(pad=0.5)
            plotname = dirname + '/' + \
                'Context_' + context_code + '.png'
            plt.savefig(plotname)


def make_separate_plots(all_logdirs,
                        legend=None,
                        xaxis=None,
                        values=None,
                        count=False,
                        font_scale=1.5,
                        smooth=1,
                        select=None,
                        exclude=None,
                        estimator='mean',
                        per_seed_comp=False):

    phases = os.listdir(all_logdirs[0])

    if ('training' in phases):

        all_training_logdirs = all_logdirs[0] + '/training/'
        exp_dir = os.listdir(all_training_logdirs)

        for exp in exp_dir:

            all_training_exps = all_training_logdirs + exp

            train_data, distr_data = get_all_datasets([all_training_exps],
                                                      legend, select, exclude)

            timesteps = train_data[0]['Epoch'].values
            average_returns = train_data[0]['AverageEpRet'].values

            plt.figure()
            plt.plot(timesteps, average_returns)
            plt.title('Average Rewards')
            plt.xlabel('Epoch')
            plt.ylabel('Average rewards per epoch')
            plotname = all_training_exps + '/' + 'AverageTrainRewards.png'
            plt.savefig(plotname)

            # Plot distributions
            if len(distr_data) > 0:
                plot_context_distributions(distr_data, all_training_exps)


def make_reward_per_ctx_plots(all_logdirs,
                              legend=None,
                              xaxis=None,
                              values=None,
                              count=False,
                              font_scale=1.5,
                              smooth=1,
                              select=None,
                              exclude=None,
                              estimator='mean',
                              ctx_width=1):

    phases = os.listdir(all_logdirs[0])
    print('Plotting rewards per context...')

    if ('training' in phases):

        # Look for file contexts.txt
        all_training_logdirs = all_logdirs[0] + '/training/'
        exp_dir = os.listdir(all_training_logdirs)

        for exp in exp_dir:

            train_path = all_training_logdirs + exp

            files = os.listdir(train_path)

            if 'contexts.txt' in files:
                ctx_data = pd.read_table(
                    os.path.join(train_path, 'contexts.txt'))

            # progress_data = pd.read_table(
            #     os.path.join(train_path, 'progress.txt'))

            # convert strings to floats (some of our old files have
            # 'tensor(0.0)' in the rewards)
            extract_num = re.compile(r"[-+]?\d*\.\d+|\d+")

            def get_float(x):
                return float(extract_num.findall(str(x))[0])

            ctx_data['Final reward'] = ctx_data['Final reward'].map(get_float)

            # convert to numpy arrays
            contexts = ctx_data.loc[
                :, ctx_data.columns.str.startswith('Context')].values
            rewards = ctx_data['Final reward'].values

            # find repeated rewards and remove them
            select = np.ones(len(rewards), dtype=np.bool)
            select[1:] = rewards[1:] != rewards[:-1]
            rewards = rewards[select]
            contexts = contexts[select]

            t = range(contexts.shape[0])

            for d in range(contexts.shape[-1]):
                plt.figure()
                plt.title('Average Rewards')
                x, y = contexts[:, d], rewards,
                plt.scatter(x, y, c=t, alpha=0.9)
                plt.xlabel('Context')
                plt.ylabel('Average rewards per epoch')
                plt.colorbar()

                plotname = train_path + '/' + 'BestRewardsPerCtxs_%d.png' % (d)
                plt.savefig(plotname)

    print('Done.')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--srcdir', nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='Iteration')
    parser.add_argument(
        '--value',
        '-y',
        default='Average Cumulative Reward',
        nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    parser.add_argument('--style', default='separate')
    parser.add_argument('--root', default='data')
    parser.add_argument('--fixed_mean', action='store_true')
    parser.add_argument('--vary_mean', action='store_true')
    args = parser.parse_args()
    """

    Args:
        logdir (strings): As many log directories (or prefixes to log
            directories, which the plotter will autocomplete internally) as
            you'd like to plot from.

        legend (strings): Optional way to specify legend for the plot. The
            plotter legend will automatically use the ``exp_name`` from the
            config.json file, unless you tell it otherwise through this flag.
            This only works if you provide a name for each directory that
            will get plotted. (Note: this may not be the same as the number
            of logdir args you provide! Recall that the plotter looks for
            autocompletes of the logdir args: there may be more than one
            match for a given logdir prefix, and you will need to provide a
            legend string for each one of those matches---unless you have
            removed some of them as candidates via selection or exclusion
            rules (below).)

        xaxis (string): Pick what column from data is used for the x-axis.
             Defaults to ``TotalEnvInteracts``.

        value (strings): Pick what columns from data to graph on the y-axis.
            Submitting multiple values will produce multiple graphs. Defaults
            to ``Performance``, which is not an actual output of any algorithm.
            Instead, ``Performance`` refers to either ``AverageEpRet``, the
            correct performance measure for the on-policy algorithms, or
            ``AverageTestEpRet``, the correct performance measure for the
            off-policy algorithms. The plotter will automatically figure out
            which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for
            each separate logdir.

        count: Optional flag. By default, the plotter shows y-values which
            are averaged across all results that share an ``exp_name``,
            which is typically a set of identical experiments that only vary
            in random seed. But if you'd like to see all of those curves
            separately, use the ``--count`` flag.

        smooth (int): Smooth data by averaging it over a fixed window. This
            parameter says how wide the averaging window will be.

        select (strings): Optional selection rule: the plotter will only show
            curves from logdirs that contain all of these substrings.

        exclude (strings): Optional exclusion rule: plotter will only show
            curves from logdirs that do not contain these substrings.

    """

    if args.style == 'separate':

        make_separate_plots(
            args.logdir,
            args.legend,
            args.xaxis,
            args.value,
            args.count,
            smooth=args.smooth,
            select=args.select,
            exclude=args.exclude,
            estimator=args.est)

    elif args.style == 'reward-per-ctx':

        make_reward_per_ctx_plots(
            args.logdir,
            args.legend,
            args.xaxis,
            args.value,
            args.count,
            smooth=args.smooth,
            select=args.select,
            exclude=args.exclude,
            estimator=args.est)
    else:
        raise ValueError('Invalid plot style')


if __name__ == "__main__":
    main()
