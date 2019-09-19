#!/usr/bin/env python
import argparse
import json
import numpy as np
import torch
import tqdm
from torch import nn

from lsdr.utils.create_experiments import deserialize_arrays


class Discriminator(nn.Module):
    def __init__(self, idims):
        super(Discriminator, self).__init__()
        hdims = 128
        model = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(idims, hdims)),
            nn.ELU(),
            nn.utils.spectral_norm(nn.Linear(hdims, hdims)),
            nn.ELU(),
            nn.utils.spectral_norm(nn.Linear(hdims, 1)),
        )
        self.model = model

    def forward(self, inputs):
        return self.model(inputs).view(-1)


def neural_wassertein(P, Q, steps=5000, batch_size=1000, quiet=False):
    dims = P.sample().shape[0]
    model = Discriminator(dims)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    if torch.cuda.is_available:
        model = model.cuda()
    model.train()
    pbar = tqdm.tqdm(range(steps), disable=quiet)
    dist = None
    for i in pbar:
        p = P.sample(torch.Size([batch_size]))
        q = Q.sample(torch.Size([batch_size]))
        if torch.cuda.is_available:
            p = p.cuda()
            q = q.cuda()

        model.zero_grad()
        inputs = torch.cat([p, q], 0)
        outs = model(inputs)
        Ep = outs[:batch_size].mean(0)
        Eq = outs[batch_size:].mean(0)
        d = Ep - Eq

        loss = -d
        loss.backward()
        opt.step()

        a = 0.995
        if dist is None:
            dist = d.detach()
        else:
            dist = (1 - a) * d.detach() + a * dist
        pbar.set_description('{0}'.format(dist))

    return dist.detach().cpu().numpy()


def init_dist(params, dist_type='gaussian'):
    if dist_type == 'gaussian':
        mu, L = params
        if isinstance(mu, np.ndarray):
            mu = torch.tensor(mu)
        if isinstance(L, np.ndarray):
            L = torch.tensor(L)
        mu = mu.float().detach().requires_grad_()
        L = L.float().detach().requires_grad_()
        dist = torch.distributions.MultivariateNormal(mu, scale_tril=L)
        params = [mu, L]
    elif dist_type == 'uniform':
        R = torch.tensor(params).float().detach().requires_grad_()
        lo, hi = R[:, 0], R[:, 1]
        dist = torch.distributions.Uniform(lo, hi)
        params = [R]
    else:
        raise NotImplementedError
    return dist, params


def load_dist(dist_data):
    dist_type = dist_data['type']
    dist_params = deserialize_arrays(dist_data['params'])
    return init_dist(dist_params, dist_type)[0]


def distribution_distance(P, Q, distance_type='wasserstein', **kwargs):
    # TODO use other distances (e.g. total variation, or Wasserstein)
    if distance_type == 'kl':
        klpq = torch.distributions.kl.kl_divergence(
            P, Q).detach().cpu().numpy().sum()
        klqp = torch.distributions.kl.kl_divergence(
            Q, P).detach().cpu().numpy().sum()
        return 0.5 * (klpq + klqp)
    elif distance_type == 'wasserstein':
        return neural_wassertein(P, Q, **kwargs)
    else:
        print(distance_type)
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Prints an ordered list of runs, from easier (most overlap between training ans testing) to harder (least overlap). Requires being able to evaluate the log_prob of the training distributions."
    )
    parser.add_argument(
        '--distance-type',
        type=str,
        default='wasserstein',
        help="Options: kl, wasserstein")
    parser.add_argument(
        '--input-file',
        type=str,
        default='./configs/experiments.json',
        help='file containing the experiment runs to be compared')
    parser.add_argument(
        '--steps',
        type=int,
        default=4000,
        help='steps for training the wassersteein discriminator')

    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        train_exps, test_exps, train_dists, test_dists = json.load(f)

    dists = {}
    for train_key, test_key in zip(
            sorted(train_dists.keys()), sorted(test_dists.keys())):
        train_dist = load_dist(train_dists[train_key])
        test_dist = load_dist(test_dists[test_key])

        dists[train_key + ',' + test_key] = distribution_distance(
            train_dist, test_dist, args.distance_type, steps=args.steps)

    for key in sorted(dists, key=lambda x: dists[x]):
        print(key, dists[key])