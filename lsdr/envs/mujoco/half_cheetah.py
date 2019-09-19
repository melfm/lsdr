import numpy as np
import os
from xml.etree import ElementTree as ET
from gym import utils
from gym.envs.mujoco import mujoco_env


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 torso_size=0.046,
                 torso_density=1000,
                 joint_damping=0.0,
                 friction=.4,
                 experiment_id='half_cheetah_exp'):
        self.torso_size = torso_size
        self.torso_density = torso_density
        self.joint_damping = joint_damping
        self.friction = friction
        self.experiment_id = experiment_id
        self.apply_env_modifications()
        # mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

    def apply_env_modifications(self):
        path = os.path.join(os.path.dirname(__file__),
                            "assets", 'half_cheetah.xml')
        xmldoc = ET.parse(path)
        root = xmldoc.getroot()
        for geom in root.iter('geom'):
            if geom.get('name') == 'torso':
                geom.set('size', str(self.torso_size))

        #for elem in root.iterfind('worldbody/body/geom'):
        #    elem.set('density', str(self.torso_density))

        for elem in root.iterfind('worldbody/body/joint'):
            elem.set('damping', str(self.joint_damping))

        for elem in root.iterfind('default/geom'):
            elem.set('friction', str(self.friction) + ' .1 .1')

        tmppath = '/tmp/' + self.experiment_id + '.xml'
        xmldoc.write(tmppath)
        mujoco_env.MujocoEnv.__init__(self, tmppath, 4)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(
            reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
