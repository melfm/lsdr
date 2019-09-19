import numpy as np
import torch
import unittest

import spinup.envs.environment_sampler as env_sampler


class EnvSamplerTest(unittest.TestCase):

    dump_output = True

    def test_init_env_sampler(self):

        experiment_id = None
        default_env = False
        dist_params = [
            np.array([123.41343819, 19.49169346, 0.50704372]),
            np.array([[3.27582454e+00, 0.00000000e+00, 0.00000000e+00],
                      [4.60283390e-01, 1.24207294e+00, 0.00000000e+00],
                      [-4.36672822e-01, -2.94320942e-03, 2.47945398e-01]])
        ]

        dist_type = 'gaussian'

        context_sampler = env_sampler.init_env_sampler(
            'lunar-lander',
            0,
            experiment_id,
            default_env=default_env,
            init_dist_params=dist_params,
            dist_type=dist_type)

        env_test = context_sampler.sample(1)
        self.assertAlmostEqual(env_test.LEG_SPRING_TORQUE, 128.46, places=2)
        self.assertAlmostEqual(env_test.MAIN_ENGINE_POWER, 19.836, places=2)
        self.assertAlmostEqual(env_test.SIDE_ENGINE_POWER, -0.705, places=2)

    def test_hopper_env_sampler(self):

        experiment_id = 'test_0'
        default_env = False

        dist_params = [
            np.array([3.78447184, 0.02307786, 2.84803485]),
            np.array([[0.55515483, 0., 0.], [0.02575168, 0.11463837, 0.],
                      [0.01384813, 0.37316284, 0.69501602]])
        ]

        dist_type = 'gaussian'

        context_sampler = env_sampler.init_env_sampler(
            'hopper',
            seed=0,
            experiment_id=experiment_id,
            default_env=default_env,
            init_dist_params=dist_params,
            dist_type=dist_type)

        env_test = context_sampler.sample(1)

        self.assertAlmostEqual(env_test.friction, 4.639, places=2)
        self.assertAlmostEqual(env_test.torso_size, 0.0291, places=2)
        # This should remain as default as its not being randomized
        # self.assertAlmostEqual(env_test.foot_size,
        #                       0.06, places=2)
        self.assertAlmostEqual(env_test.joint_damping, 1.245, places=2)

    def test_discrete_distribution(self):

        ranges = np.array([[0., 0.1]])
        ncells = np.array(3)
        ranges_tensor = torch.tensor(ranges).float().detach()
        ncells_tensor = torch.tensor(ncells).long().detach()

        logit_params = env_sampler.create_discrete_distr_params(
            ranges_tensor, ncells_tensor)
        dist = env_sampler.Discrete(
            ncells_tensor, ranges_tensor, params=logit_params)

        exp_logits = np.array([-1.0986, -1.0986, -1.0986])
        exp_probs = np.array([0.3333, 0.3333, 0.3333])
        exp_entropy = np.array(1.0986)
        np.testing.assert_array_almost_equal(
            exp_logits, dist.logits.data.numpy(), decimal=4)
        np.testing.assert_array_almost_equal(
            exp_logits, dist._logits.data.numpy(), decimal=4)
        np.testing.assert_array_almost_equal(
            exp_probs, dist.probs.data.numpy(), decimal=4)

        np.testing.assert_array_almost_equal(
            exp_entropy, dist.entropy().data.numpy(), decimal=4)

    def test_discrete_distribution_via_cntx(self):
        # Update the distribution logits, as the optimizer would do
        params = [np.array([0., 0.1]), np.array(3)]
        exp_init_entropy = np.array(1.09857)

        context_sampler = env_sampler.init_env_sampler(
            'hopper',
            seed=0,
            experiment_id='unit-test',
            init_dist_params=params,
            test_dist_params=params,
            dist_type='discrete')

        train_logits = context_sampler.train_dist._logits

        context_sampler.params[0].data = context_sampler.params[0].data * \
            0.01*torch.arange(1.0*context_sampler.params[0].shape[-1])

        exp_train_logits = np.array([-0.0000, -0.0110, -0.0220])
        exp_train_logits_norm = np.array([-1.0877, -1.0987, -1.1096])
        exp_train_probs = np.array([0.33699554, 0.3333089, 0.32969556])
        train_logits = context_sampler.train_dist._logits.data.numpy()
        train_logits_fix = context_sampler.train_dist.logits.data.numpy()
        train_probs = context_sampler.train_dist.probs.data.numpy()

        np.testing.assert_array_almost_equal(
            exp_train_logits, train_logits, decimal=4)
        np.testing.assert_array_almost_equal(
            exp_train_logits_norm, train_logits_fix, decimal=4)
        np.testing.assert_array_almost_equal(
            exp_train_probs, train_probs, decimal=4)

        train_dist_entropy = context_sampler.train_dist.entropy().data.numpy(),
        np.testing.assert_array_almost_equal(
            exp_init_entropy, train_dist_entropy, decimal=4)

    def test_kl_div_loss(self):
        test_params = [
            np.array([-3.0, 10.0]),
            np.array([-0.1, 0.2]),
            np.array([-10.0, 30.0]),
            np.array(4)
        ]
        dist_params = [np.array([0.1, 0.1, 0.1]), np.eye(3)]

        sampler = env_sampler.init_env_sampler(
            'hopper',
            seed=0,
            experiment_id='test_kl_div_loss_0',
            init_dist_params=dist_params,
            dist_type='gaussian',
            test_dist_params=test_params)

        p_train = sampler.train_dist
        p_test = sampler.test_dist

        m, n = 5000, 500
        opt = torch.optim.Adam(sampler.params, 1e-2)
        for i in range(m):
            opt.zero_grad()
            z = p_test.sample(torch.Size([n]))
            log_p_train = p_train.log_prob(z)
            log_p_test = p_test.log_prob(z)
            kl_samples = log_p_test - log_p_train
            kl_loss = kl_samples.mean(0)
            kl_loss.mean().backward()
            opt.step()

        # if the kl divergence was optimized correctly, the mean
        # and stdev of the gaussian should be close to the test dist
        # (error at most 10% of test dist mean and standard dev)
        train_mean = p_train.mean.detach()
        train_std = (p_train._unbroadcasted_scale_tril).diag().detach()
        test_mean = 0.5 * (p_test.lo + p_test.hi)
        test_std = (p_test.hi - p_test.lo) / (12**0.5)

        assert (all(abs(train_mean - test_mean) / test_mean < 1e-1))
        assert (all(abs(train_std - test_std) / test_std < 1e-1))


if __name__ == '__main__':
    unittest.main()
