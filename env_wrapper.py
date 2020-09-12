import gym
from dm_control import suite
from dm_control.suite.wrappers import pixels
from gym.spaces import Box
import dm_env
import numpy as np
import mujoco_py



class EnvWrapper:
    GYM = "gym"
    DM_CONTROL = "dm_control"

    def __init__(self, domain, task, pixel_based=False, res=64):
        self.type = EnvWrapper.GYM if task is None else EnvWrapper.DM_CONTROL
        self.domain = domain
        self.task = task
        self.pixel_based = pixel_based
        self.resolution = res
        if self.type == EnvWrapper.GYM:
            self.env = gym.make(domain)
            self.action_space = self.env.action_space
            self.qpos_size = self.env.sim.get_state().qpos.shape[0]
            if pixel_based:
                self.env.env.viewer = mujoco_py.MjRenderContextOffscreen(self.env.env.sim, 0)
            self.max_episode_steps = self.env._max_episode_steps
        elif self.type == EnvWrapper.DM_CONTROL:
            self.env = suite.load(domain_name=domain, task_name=task)
            self.action_space = Box(low=self.env.action_spec().minimum, high=self.env.action_spec().maximum, dtype=np.float32)
            self.qpos_size = self.env.physics.data.qpos.shape[0]
            # if pixel_based:
            #     self.pixel_env = pixels.Wrapper(self.env, render_kwargs={"camera_id" : 0, "width" : res, "height": res})
            # This doesn't exist on dm_control, so let's just manually set it?
            self.max_episode_steps = 1000

    def step(self, action):
        if self.type == EnvWrapper.GYM:
            return self.env.step(action)
        elif self.type == EnvWrapper.DM_CONTROL:
            step = self.env.step(action)
            # step.observation is too funky. qpos and qvel are the only things that are reliable.
            state = self.flatten_obs(step.observation)
            reward = step.reward
            done = step.step_type is dm_env.StepType.LAST
            return state, reward, done, None

    def reset(self):
        if self.type == EnvWrapper.GYM:
            return self.env.reset()
        elif self.type == EnvWrapper.DM_CONTROL:
            return self.flatten_obs(self.env.reset().observation)

    def get_current_state(self, temp_state=None, position_only=False):
        if temp_state is None:
            temp_state = self.reset()

        if self.type == EnvWrapper.GYM:
            # We need the flattened states for pixel based training
            if self.pixel_based:
                return self.env.sim.get_state().flatten()
            elif position_only:
                return self.env.sim.get_state().qpos
            # The normal states
            else:
                return temp_state
        elif self.type == EnvWrapper.DM_CONTROL:
            if self.pixel_based:
                return np.concatenate((self.env.physics.data.qpos, self.env.physics.data.qvel))
            elif position_only:
                return self.env.physics.qpos
            # The flatten argument does not matter here, only for gym environments
            else:
                return temp_state

    def flattened_to_pixel(self, flattened):
        if self.type == EnvWrapper.GYM:
            # Set the state and render the pixel image
            self.env.sim.set_state_from_flattened(flattened)
            img = self.env.env.sim.render(camera_name='track', width=self.resolution, height=self.resolution,
                                                depth=False)
            # scale between -1/2 and 1/2
            return (img.transpose((2, 0, 1)) / 255) - 0.5
        elif self.type == EnvWrapper.DM_CONTROL:
            qpos_n = flattened[:self.qpos_size]
            qvel_n = flattened[self.qpos_size:]
            # Set the state to the position/velocity
            with self.env.physics.reset_context():
                self.env.physics.data.qpos[:] = qpos_n  # Set  position ,
                self.env.physics.data.qvel[:] = qvel_n  # velocity
            # Then render the image
            img = self.env.physics.render(camera_id=0, width=self.resolution, height=self.resolution)
            # Transpose dimensions and scale between -1/2 and 1/2
            return (img.transpose((2, 0, 1)) / 255) - 0.5

    def get_state_before_eval(self):
        if self.type == EnvWrapper.GYM:
            return self.env.sim.get_state()
        elif self.type == EnvWrapper.DM_CONTROL:
            return (self.env.physics.data.qpos,
                    self.env.physics.data.qvel,
                    self.env.physics.data.ctrl)

    def set_state_after_eval(self, before_eval_state):
        if self.type == EnvWrapper.GYM:
            self.env.set_state(before_eval_state)
        elif self.type == EnvWrapper.DM_CONTROL:
            with self.env.physics.reset_context():
                self.env.physics.data.qpos[:] = before_eval_state[0]
                self.env.physics.data.qvel[:] = before_eval_state[1]
                self.env.physics.data.ctrl[:] = before_eval_state[2]

    def get_state_space_size(self, position_only=False):
        if self.type == EnvWrapper.GYM:
            if self.pixel_based:
                return self.env.sim.get_state().flatten().shape[0]
            elif position_only:
                return self.env.sim.get_state().qpos.shape[0]
            else:
                return self.env.reset().shape[0]
        elif self.type == EnvWrapper.DM_CONTROL:
            if self.pixel_based:
                return self.env.physics.data.qpos.shape[0] + self.env.physics.data.qvel.shape[0]
            elif position_only:
                return self.env.physics.data.qpos.shape[0]
            else:
                return self.flatten_obs(self.env.task.get_observation(self.env.physics)).shape[0]

    def flatten_obs(self, observation):
        return np.hstack([np.array([observation[x]]).flatten() for x in observation])

    def close(self):
        # Same syntax in both situations
        self.env.close()

    def seed(self, sd):
        if self.type == EnvWrapper.GYM:
            self.env.seed(sd)
        elif self.type == EnvWrapper.DM_CONTROL:
            pass # TODO: Figure out how to set the seed

