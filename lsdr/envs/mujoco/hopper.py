import numpy as np
import os
from xml.etree import ElementTree as ET
from gym import utils
from gym.envs.mujoco import mujoco_env


class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, foot_friction=2.0,
                 torso_size=0.05,
                 torso_density=1000,
                 joint_damping=0,
                 experiment_id='hopper_exp'):
        self.friction = foot_friction
        self.torso_size = torso_size
        self.torso_density = torso_density
        self.joint_damping = joint_damping
        self.experiment_id = experiment_id
        self.apply_env_modifications()
        # mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)

    def apply_env_modifications(self):
        path = os.path.join(os.path.dirname(__file__), "assets", 'hopper.xml')
        xmldoc = ET.parse(path)
        root = xmldoc.getroot()

        for elem in root.iterfind('worldbody/body/geom'):
            elem.set('density', str(self.torso_density))

        for geom in root.iter('geom'):
            if geom.get('name') == 'torso_geom':
                # print('torso size =', self.torso_size)
                geom.set('size', str(self.torso_size))
            if geom.get('name') == 'foot_geom':
                # print('foot friction =', self.friction)
                geom.set('friction', str(self.friction))
                # geom.set('size', str(self.foot_size))

        for joint in root.iter('joint'):
            damping = joint.get('damping')
            # Only change the joint damping defined inside body
            # which by default are set to 0
            if damping == '0':
                joint.set('damping', str(self.joint_damping))

        tmppath = '/tmp/' + self.experiment_id + '.xml'
        xmldoc.write(tmppath)
        mujoco_env.MujocoEnv.__init__(self, tmppath, 4)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(
            low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
