import numpy as np
import torch

from functools import partial

from lsdr.envs.classic_control import cartpole, mountain_car
from lsdr.envs.box2d import lunar_lander
from lsdr.envs.mujoco import (hopper, cartpole as mj_cartpole, half_cheetah,
                              humanoid)

available_envs = [
    'cartpole', 'lunar-lander', 'mountain-car',
    'cartpole-swingup', 'cartpole-swingup-pl', 'cartpole-swingup-pm',
    'cartpole-swingup-cm', 'cartpole-swingup-cd', 'cartpole-swingup-pole-only',
    'humanoid', 'hopper', 'hopper-torso-only', 'hopper-density-only',
    'hopper-damping-only', 'hopper-friction-only', 'half-cheetah',
    'half-cheetah-torso', 'half-cheetah-density', 'half-cheetah-friction',
    'half-cheetah-damping'
]


class Delta(torch.distributions.Uniform):
    def __init__(self, value, validate_args=None):
        return super().__init__(value, value, validate_args=validate_args)


class Discrete(torch.distributions.Categorical):
    def __init__(self, cells_per_dim, ranges, params, validate_args=None):
        """Implements context distribution with a discrete (categorical
        distribution). To enable learning the distribution parameters,
        we override the @lazy_property attributes of the Categorical
        super class.
        """
        # parameters defining the support of this distribution
        self.lo = ranges[:, 0]
        self.hi = ranges[:, 1]
        self.widths = self.hi - self.lo

        # parameters defining the cells per context dimension
        self.ncells = cells_per_dim
        if self.ncells.numel() == 1:
            self.ncells = self.ncells.expand_as(self.lo)

        # Helper variable for converting multidimensional coordinates to
        # linear indices, since the underlying categorical distribution is 1D
        self.cumprods = torch.cat(
            [torch.tensor([1]),
             self.ncells.cumprod(-1)[:-1].long()], -1)

        self._logits = params[0]
        super().__init__(logits=params[0], validate_args=validate_args)

    @property
    def logits(self):
        # ensure the logits correspond to a valid discrete distribution
        # (probs sum to one)
        return self._logits - self._logits.logsumexp(dim=-1, keepdim=True)

    @logits.setter
    def logits(self, x):
        # only copy the data, don't replace self._logits (since we want to
        # update them with an optimizer)
        self._logits.data = x.data

    @property
    def probs(self):
        # use the current logits to compute the probs
        return torch.distributions.utils.logits_to_probs(self.logits)

    def to_category(self, values):
        # transform incoming values into cell coordinates
        coords = (self.ncells.float() * (values - self.lo) /
                  self.widths).floor().long()
        # ensure the coords do not exceed the indices per dimension
        coords = torch.min(coords, self.ncells - 1)
        # convert coordinates into category indices
        idxs = (self.cumprods * coords).sum(-1)
        return idxs

    def from_category(self, categories):
        # convert flat indices to coordinates
        coords = torch.tensor(
            np.unravel_index(categories,
                             self.ncells.numpy().tolist(), 'F')).t().float()

        # convert coords into samples from each cell
        # adding U[0,1] samples to get values other than the
        # lower limits of a cell
        u = torch.rand_like(coords)
        values = (coords + u) * self.widths / self.ncells.float() + self.lo
        return values

    def sample(self, sample_shape=torch.Size()):
        cat_samples = super(Discrete, self).sample(sample_shape)
        if len(sample_shape) == 0:
            return self.from_category([cat_samples])[0]
        return self.from_category(cat_samples)

    def log_prob(self, value):
        return super(Discrete, self).log_prob(self.to_category(value))


class SafeMultivariateNormal(torch.distributions.MultivariateNormal):
    def __init__(self,
                 loc,
                 covariance_matrix=None,
                 precision_matrix=None,
                 scale_tril_offdiag=None,
                 log_scale_tril_diag=None,
                 validate_args=None,
                 ranges=None):
        D = scale_tril_offdiag.size(0)
        self.scale_tril_offdiag = scale_tril_offdiag
        self.log_scale_tril_diag = log_scale_tril_diag
        if ranges is not None:
            self.lo = ranges[:, 0]
            self.hi = ranges[:, 1]
            self.widths = (self.hi - self.lo) / (12**0.5)
            self.center = (self.lo + self.hi) / 2.0

            L_off = self.scale_tril_offdiag
            log_L_diag = self.log_scale_tril_diag
            self.scale_tril_offdiag.data = (L_off.t() / self.widths).t()
            self.log_scale_tril_diag.data = log_L_diag - torch.log(self.widths)

        scale_tril = self.scale_tril_offdiag + torch.eye(D) * torch.exp(
            self.log_scale_tril_diag)

        super(SafeMultivariateNormal, self).__init__(
            loc,
            covariance_matrix=covariance_matrix,
            precision_matrix=precision_matrix,
            scale_tril=scale_tril,
            validate_args=validate_args)

        if ranges is not None:
            self._loc = loc
            self._loc.data = (self._loc.detach() - self.center) / self.widths
            self.update_scale_tril()
            self.scale_tril = self._unbroadcasted_scale_tril

    @property
    def loc(self):
        if hasattr(self, 'widths'):
            return self._loc * self.widths + self.center
        else:
            return self._loc

    @loc.setter
    def loc(self, loc):
        if hasattr(self, 'widths'):
            self._loc = (loc - self.center) / self.widths
        else:
            self._loc = loc

    def update_scale_tril(self):
        D = self.scale_tril_offdiag.size(0)
        mask = torch.ones(D, D).tril(-1)
        L = (self.scale_tril_offdiag * mask +
             torch.eye(D) * torch.exp(self.log_scale_tril_diag))
        if hasattr(self, 'widths'):
            L = ((L.t()) * self.widths).t()
        self._unbroadcasted_scale_tril = L
        self.scale_tril.data = L.data

    def sample(self, *args, **kwargs):
        self.update_scale_tril()
        return super(SafeMultivariateNormal, self).sample(*args, **kwargs)

    def log_prob(self, value):
        self.update_scale_tril()
        return super(SafeMultivariateNormal, self).log_prob(value)

    def entropy(self):
        self.update_scale_tril()
        return super(SafeMultivariateNormal, self).entropy()


class EnvSampler(object):
    """Env Class used for sampling environments with different
        simulation parameters, from a distribution specified by
        self.dist
    """

    def __init__(self,
                 env_constructor,
                 dist,
                 params,
                 seed,
                 env_name,
                 test_dist=None):
        assert isinstance(dist, torch.distributions.Distribution)
        if test_dist:
            assert isinstance(test_dist, torch.distributions.Distribution)
        self.train_dist = dist
        self.test_dist = test_dist
        self.env_constructor = env_constructor
        self.params = params
        self.seed = seed
        torch.manual_seed(seed)
        self.env_name = env_name

    def parameters(self):
        return self.params

    def sample(self, n_samples=1, return_ctxs=False, quiet=True,
               distr='train'):

        if distr == 'train':
            ctxs = self.train_dist.sample(torch.Size([n_samples]))
        elif distr == 'test':
            ctxs = self.test_dist.sample(torch.Size([n_samples]))
        else:
            raise ValueError('Invalid distribution type.')
        envs = []
        for ctx in ctxs:
            ctx_ = ctx.cpu().numpy()
            try:
                env = self.env_constructor(*ctx_.tolist())
                env.seed(self.seed)
                setattr(env, 'context', ctx)
            except Exception:
                if not quiet:
                    import traceback
                    traceback.print_exc()
                env = None
            envs.append(env)

        if len(envs) == 1:
            envs = envs[0]
            ctxs = ctxs[0]

        if return_ctxs:
            return envs, ctxs
        else:
            return envs

    def get_observation_space(self, cat_context=False, full_dim=False):
        env = None
        while env is None:
            env = self.sample()
        obs_dim = env.observation_space.shape

        if cat_context:
            ctx = env.context
            if full_dim:
                obs_dim = list(obs_dim)
                obs_dim[0] += len(ctx)
                obs_dim = tuple(obs_dim)
                return obs_dim
            return obs_dim, len(ctx)
        else:
            return obs_dim


def sample_env(context_sampler, distr='train', env_steps=5):
    valid_env = False
    # Keep trying until a valid env is found.
    while not valid_env:
        try:
            env = context_sampler.sample(distr=distr)
            if env is not None:
                env.reset()
                for i in range(env_steps):
                    env.step(env.action_space.sample())
                valid_env = True
        except Exception:
            import traceback
            traceback.print_exc()
            valid_env = False
    return env


def reacher_constructor(z1):

    return reacher.Reacher(mass=z1)


def lunar_lander_constructor(z1, z2, z3):
    return lunar_lander.LunarLander(
        leg_spring_torque=z1, main_engine_power=z2, side_engine_power=z3)


def cartpole_swingup_constructor(z1, z2, z3, z4):
    return mj_cartpole.Cartpole(
        cart_mass=z1, pole_mass=z2, pole_length=z3, cart_damping=z4)


def cartpole_swingup_pole_only_constructor(z1, z2):
    return mj_cartpole.Cartpole(pole_mass=z1, pole_length=z2)


def cartpole_swingup_cart_mass_constructor(z1):
    return mj_cartpole.Cartpole(cart_mass=z1)


def cartpole_swingup_pole_mass_constructor(z2):
    return mj_cartpole.Cartpole(pole_mass=z2)


def cartpole_swingup_pole_length_constructor(z3):
    return mj_cartpole.Cartpole(pole_length=z3)


def cartpole_swingup_cart_damp_constructor(z4):
    return mj_cartpole.Cartpole(cart_damping=z4)


def hopper_constructor(z1, z2, z3, experiment_id=None):

    return hopper.HopperEnv(
        foot_friction=z1,
        torso_size=z2,
        joint_damping=z3,
        experiment_id=experiment_id)


def hopper_torso_only_constructor(z1, experiment_id=None):

    return hopper.HopperEnv(torso_size=z1, experiment_id=experiment_id)


def hopper_density_only_constructor(z1, experiment_id=None):

    return hopper.HopperEnv(torso_density=z1, experiment_id=experiment_id)


def hopper_damping_only_constructor(z1, experiment_id=None):

    return hopper.HopperEnv(joint_damping=z1, experiment_id=experiment_id)


def hopper_friction_only_constructor(z1, experiment_id=None):

    return hopper.HopperEnv(foot_friction=z1, experiment_id=experiment_id)


def half_cheetah_constructor(z1, z2, z3, experiment_id=None):

    return half_cheetah.HalfCheetahEnv(
        friction=z1,
        torso_size=z2,
        joint_damping=z3,
        experiment_id=experiment_id)


def half_cheetah_torso_only_constructor(z1, experiment_id=None):

    return half_cheetah.HalfCheetahEnv(
        torso_size=z1, experiment_id=experiment_id)


def half_cheetah_density_constructor(z1, experiment_id=None):

    return half_cheetah.HalfCheetahEnv(
        torso_density=z1, experiment_id=experiment_id)


def half_cheetah_damping_only_constructor(z1, experiment_id=None):

    return half_cheetah.HalfCheetahEnv(
        joint_damping=z1, experiment_id=experiment_id)


def half_cheetah_friction_constructor(z1, experiment_id=None):

    return half_cheetah.HalfCheetahEnv(
        friction=z1, experiment_id=experiment_id)


def humanoid_constructor(z1, z2, z3):
    return humanoid.HumanoidEnv(wind=[z1, z1], gravity=z2, air_viscosity=z3)


def init_dist(params,
              test_dist_params=None,
              dist_type='gaussian',
              rescale=True):
    if dist_type == 'gaussian':
        mu, L = params
        if isinstance(mu, np.ndarray):
            mu = torch.tensor(mu)
        if isinstance(L, np.ndarray):
            L = torch.tensor(L)
        mu = mu.float().detach()
        L = L.float().detach()
        D = L.size(0)
        mask = torch.ones(D, D).tril(-1).byte()
        L_off = torch.zeros(D, D).float()
        L_off[mask] = L[mask]
        log_L_diag = torch.log(L.diag())

        mu.requires_grad_(True)
        L_off.requires_grad_(True)
        log_L_diag.requires_grad_(True)

        # initialize distribution
        test_dist = None
        rngs = None
        if test_dist_params is not None:
            # We have the case of guassian train distr and uniform test
            # at least for now assume that is the case (because it could
            # also be just gaussian ...
            test_ranges, test_ncells = \
                test_dist_params[:-1], test_dist_params[-1]
            test_ranges_tensor = torch.tensor(
                test_ranges).float().detach().squeeze(0)
            test_ncells_tensor = torch.tensor(test_ncells).long()
            test_params = create_discrete_distr_params(test_ranges_tensor,
                                                       test_ncells_tensor)
            test_dist = Discrete(
                test_ncells_tensor, test_ranges_tensor, params=test_params)

            if rescale:
                rngs = test_ranges_tensor

        dist = SafeMultivariateNormal(
            mu,
            scale_tril_offdiag=L_off,
            log_scale_tril_diag=log_L_diag,
            ranges=rngs)

        train_params = [mu, L_off, log_L_diag]
        return dist, train_params, test_dist

    elif dist_type == 'uniform':

        if len(params) >= 2:
            # Need this hack because `discrete` distr will try a uniform
            # distr for during transfer and its forcing dist_type of
            # uniform to enter here.
            params = params[:-1]
        # Create two distributions, one for train and one for test
        R = torch.tensor(params).float().detach().requires_grad_()
        lo, hi = R[:, 0], R[:, 1]
        train_dist = torch.distributions.Uniform(lo, hi)
        params = [R]
        # The parameter coming in, is designed for discrete distr
        # So either do a good fix, or take the first range and
        # ignore the bin parameter.
        range_test = torch.tensor([test_dist_params[:-1]]).float()
        if range_test.shape[0] == 1:
            range_test = torch.squeeze(range_test, dim=0)
        lo, hi = range_test[:, 0], range_test[:, 1]
        test_dist = torch.distributions.Uniform(lo, hi)
        return train_dist, params, test_dist

    elif dist_type == 'delta':
        context = torch.tensor(params).flatten().requires_grad_()
        dist = Delta(context)
        train_params = [context]

    elif dist_type == 'discrete':
        ranges, ncells = params[:-1], params[-1]
        ranges_tensor = torch.tensor(ranges).float().detach()
        # Why are these type conversions needed?
        # Who knows, this is how pytorch does it.
        train_ncells_tensor = torch.tensor(ncells).long()
        train_params = create_discrete_distr_params(ranges_tensor,
                                                    train_ncells_tensor)
        dist = Discrete(
            train_ncells_tensor, ranges_tensor, params=train_params)

        # Create test distribution
        if test_dist_params is None:
            raise ValueError(
                'Test distribution must be defined for discrete distr.')
        test_ranges, test_ncells = test_dist_params[:-1], test_dist_params[-1]
        test_ranges_tensor = torch.tensor(test_ranges).float().detach()
        test_ncells_tensor = torch.tensor(test_ncells).long()
        test_params = create_discrete_distr_params(test_ranges_tensor,
                                                   test_ncells_tensor)
        test_dist = Discrete(
            test_ncells_tensor, test_ranges_tensor, params=test_params)

        return dist, train_params, test_dist

    return dist, params, None


def create_discrete_distr_params(ranges_tensor, ncells_tensor):

    if ncells_tensor.numel() == 1:
        ncells_tensor = ncells_tensor.expand_as(ranges_tensor[:, 0])
    probs = torch.ones(*ncells_tensor).flatten()
    probs = probs / probs.sum()
    logit_params = [
        torch.distributions.utils.probs_to_logits(probs).requires_grad_()
    ]
    return logit_params


def create_def_gaussian(mean_array, noise_std):

    mu = torch.tensor(mean_array)
    S = torch.rand(mu.shape[-1], mu.shape[-1])
    S = noise_std * S.mm(S.t()) + noise_std * mu.detach() * torch.eye(
        S.shape[0])
    L = torch.cholesky(S).clone().detach()
    init_dist_params = mu, L

    return init_dist_params


def init_env_sampler(env_name,
                     seed=None,
                     experiment_id=None,
                     default_env=False,
                     init_dist_params=None,
                     test_dist_params=[[0.0, 0.1], 100],
                     dist_type='gaussian',
                     rescale=True):
    if env_name == "cartpole":
        # the context distribution corresponds to the cart mass, pole mass and
        # pole length
        if dist_type == 'gaussian' and init_dist_params is None:
            init_dist_params = create_def_gaussian([1.0, 0.1, 0.5], 1e-1)

        env_constructor = cartpole.Cartpole

    elif env_name == "reacher1D":
        if dist_type == 'gaussian' and init_dist_params is None:
            init_dist_params = create_def_gaussian([1.0], 1e-1)

        env_constructor = reacher_constructor

    elif env_name == 'lunar-lander':
        if dist_type == 'gaussian' and init_dist_params is None:
            init_dist_params = create_def_gaussian([120.0, 20.0, 0.6], 1e-1)

        env_constructor = lunar_lander_constructor if not default_env \
            else lunar_lander.LunarLander

    elif env_name == 'mountain-car':
        # the conxtext distribution corresponds to min_position, max_position,
        # max_speed, goal_position, thrust magnitude and gravity
        if dist_type == 'gaussian' and init_dist_params is None:
            # Is the 1e-2 important?
            # S = 1e-1 * S.mm(S.t()) + 1e-2 * ...
            init_dist_params = create_def_gaussian(
                [1.2, 0.6, 0.07, 0.001, -0.0025], 1e-1)
        env_constructor = mountain_car.MountainCarEnv

    elif env_name.startswith('cartpole-swingup-pole-only'):
        env_constructor = cartpole_swingup_pole_only_constructor
        default_params = [0.5, 1.0]
        if dist_type == 'gaussian' and init_dist_params is None:
            init_dist_params = create_def_gaussian(default_params, 1e-2)

    elif env_name.startswith('cartpole-swingup'):
        if env_name.endswith('-pl'):
            env_constructor = cartpole_swingup_pole_length_constructor
            idx = slice(0, 1)
        elif env_name.endswith('-pm'):
            env_constructor = cartpole_swingup_pole_mass_constructor
            idx = slice(1, 2)
        elif env_name.endswith('-cm'):
            env_constructor = cartpole_swingup_cart_mass_constructor
            idx = slice(2, 3)
        elif env_name.endswith('-cd'):
            env_constructor = cartpole_swingup_cart_damp_constructor
            idx = slice(3, 4)
        else:
            env_constructor = cartpole_swingup_constructor
            idx = slice(0, 4)

        default_params = [0.5, 0.5, 1.0, 0.1]
        if dist_type == 'gaussian' and init_dist_params is None:
            init_dist_params = create_def_gaussian(default_params[idx], 1e-2)

    elif env_name == 'hopper':
        # Context order : foot friction, torso size, joint damping
        if dist_type == 'gaussian' and init_dist_params is None:
            init_dist_params = create_def_gaussian([2.0, 0.05, 1.0], 1e-1)

        env_constructor = partial(
            hopper_constructor, experiment_id=experiment_id
        ) if not default_env else hopper.HopperEnv()

    elif env_name == 'hopper-torso-only':
        # Context is only torso size
        if dist_type == 'gaussian' and init_dist_params is None:
            init_dist_params = create_def_gaussian([0.05], 1e-1)

        env_constructor = partial(
            hopper_torso_only_constructor, experiment_id=experiment_id
        ) if not default_env else hopper.HopperEnv()

    elif env_name == 'hopper-density-only':
        # Context is only torso size
        if dist_type == 'gaussian' and init_dist_params is None:
            init_dist_params = create_def_gaussian([1000.0], 1e-1)

        env_constructor = partial(
            hopper_density_only_constructor, experiment_id=experiment_id
        ) if not default_env else hopper.HopperEnv()

    elif env_name == 'hopper-damping-only':
        # Context is only damping
        if dist_type == 'gaussian' and init_dist_params is None:
            init_dist_params = create_def_gaussian([0.0], 1e-1)

        env_constructor = partial(
            hopper_damping_only_constructor, experiment_id=experiment_id
        ) if not default_env else hopper.HopperEnv()

    elif env_name == 'hopper-friction-only':
        # Context is only friction
        if dist_type == 'gaussian' and init_dist_params is None:
            init_dist_params = create_def_gaussian([2.0], 1e-1)

        env_constructor = partial(
            hopper_friction_only_constructor, experiment_id=experiment_id
        ) if not default_env else hopper.HopperEnv()

    elif env_name == 'half-cheetah':
        # Context order : torso_size, joint damping, friction
        if dist_type == 'gaussian' and init_dist_params is None:
            init_dist_params = create_def_gaussian([0.046, 0.01, 0.0], 1e-1)

        env_constructor = partial(
            half_cheetah_constructor, experiment_id=experiment_id
        ) if not default_env else half_cheetah.HalfCheetahEnv()

    elif env_name == 'half-cheetah-torso':
        if dist_type == 'gaussian' and init_dist_params is None:
            init_dist_params = create_def_gaussian([0.046], 1e-1)

        env_constructor = partial(
            half_cheetah_torso_only_constructor, experiment_id=experiment_id
        ) if not default_env else half_cheetah.HalfCheetahEnv()

    elif env_name == 'half-cheetah-density':
        if dist_type == 'gaussian' and init_dist_params is None:
            init_dist_params = create_def_gaussian([1000.0], 1e-1)

        env_constructor = partial(
            half_cheetah_density_constructor, experiment_id=experiment_id
        ) if not default_env else half_cheetah.HalfCheetahEnv()

    elif env_name == 'half-cheetah-damping':
        if dist_type == 'gaussian' and init_dist_params is None:
            init_dist_params = create_def_gaussian([0.0], 1e-1)

        env_constructor = partial(
            half_cheetah_damping_only_constructor, experiment_id=experiment_id
        ) if not default_env else half_cheetah.HalfCheetahEnv()

    elif env_name == 'half-cheetah-friction':
        if dist_type == 'gaussian' and init_dist_params is None:
            init_dist_params = create_def_gaussian([0.4], 1e-1)

        env_constructor = partial(
            half_cheetah_friction_constructor, experiment_id=experiment_id
        ) if not default_env else half_cheetah.HalfCheetahEnv()

    elif env_name == 'humanoid':
        if dist_type == 'gaussian' and init_dist_params is None:
            init_dist_params = create_def_gaussian([40, 9.8, 0.1], 1e-1)

        env_constructor = humanoid_constructor
    else:
        raise ValueError('Environment not supported!')

    dist, init_dist_params, test_dist = init_dist(
        init_dist_params, test_dist_params, dist_type, rescale=rescale)
    init_dist_params = list(init_dist_params)

    return EnvSampler(env_constructor, dist, init_dist_params, seed, env_name,
                      test_dist)
