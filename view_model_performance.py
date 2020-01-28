import argparse

import gym
import torch
import json
import time
from arrl import ARRL

ENV_NAME = "" # Put something else here.
ENV_NAME = "LunarLanderContinuous-v2"
USE_PREV_ACTION = False

parser = argparse.ArgumentParser(description='PyTorch AutoRegressiveFlows-RL Args')
args = parser.parse_args()
with open('models/' + ENV_NAME + '_parser_args_' + str(USE_PREV_ACTION) + '.txt', 'r') as f:
    args.__dict__ = json.load(f)

env = gym.make(args.env_name)

state = env.reset()
prev_action = torch.zeros(env.action_space.sample().shape)
agent = ARRL(env.observation_space.shape[0], env.action_space, args)
agent.load_model("models/" + args.env_name + "_actor_" + str(USE_PREV_ACTION) + ".model",
                 "models/" + args.env_name + "_critic_" + str(USE_PREV_ACTION) + ".model")

for _ in range(1000):
    env.render()
    action = agent.select_action(state, prev_action, eval=True)
    next_state, reward, done, _ = env.step(action)
    if done:
        #break
        pass
    prev_action = action
    state = next_state

time.sleep(1)
env.close()