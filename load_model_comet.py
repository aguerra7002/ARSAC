from comet_ml.api import API
import argparse
import gym
import numpy as np
from arrl import ARRL

# comet
api_key = 'tHDbEydFQGW7F1MWmIKlEvrly'

workspace = 'aguerra'
project_name = 'arsac-test'
comet_api = API(api_key=api_key)

# PUT THE NAME OF THE FILE WE WANT TO SAVE HERE
actor_filename = "actor_eval_1.model"
critic_filename = "critic_eval_1.model"
# PUT THE EXPERIMENT KEY HERE
experiment_id = "0e4d402ea88a4ab8b490d76b412bf006"

experiment = comet_api.get_experiment(project_name=project_name,
                                                  workspace=workspace,
                                                  experiment=experiment_id)
asset_list = experiment.get_asset_list()

# First setup the arguments
args_asset_id = [x for x in asset_list if x['fileName'] == "args"][0]['assetId']
args_dict = experiment.get_asset(args_asset_id, return_type="json")
parser = argparse.ArgumentParser(description='PyTorch AutoRegressiveFlows-RL Args')
args = parser.parse_args()
args.__dict__ = args_dict

# Next we setup the environment
env = gym.make(args.env_name)
if args.pixel_based:
    import mujoco_py
    env.env.viewer = mujoco_py.MjViewer(env.env.sim)

# Action Space Size
action_space_size = env.action_space.sample().shape[0]

# Figure out what the state size is
if args.pixel_based:
    state_space_size = 3 # Corresponding to number of channels. Maybe change to be more adaptable?
elif args.position_only:
    state_space_size = env.sim.get_state().qpos.shape[0]
else:
    state_space_size = env.reset().shape[0]


# Now initialize the agent
agent = ARRL(state_space_size, env.action_space, args)

# Now we load in the model weights from Comet
actor_asset_id = [x for x in asset_list if actor_filename == x['fileName']][0]['assetId']
critic_asset_id = [x for x in asset_list if critic_filename == x['fileName']][0]['assetId']
actor = experiment.get_asset(actor_asset_id)
critic = experiment.get_asset(critic_asset_id)
act_path = 'tmploaded/actor.model'
crt_path = 'tmploaded/critic.model'
with open(act_path, 'wb+') as f:
    f.write(actor)
with open(crt_path, 'wb+') as f:
    f.write(critic)
agent.load_model(act_path, crt_path)

# Allows us to get state
def get_state(temp_state=None):

    if temp_state is None:
        ret = env.reset() # Will reset the environment
    else:
        ret = temp_state

    if args.pixel_based:
        ret = env.env.sim.render(camera_name='track', width=args.resolution, height=args.resolution, depth=False)
        ret = ret.reshape((ret.shape[2], ret.shape[1], ret.shape[0]))
    elif args.position_only:
        ret = env.sim.get_state().qpos  # Just the position

    # Return the state, adding noise if args say we should
    return ret + (np.random.normal(0, 0.1, state_space_size) if args.add_state_noise else 0)

# Start of the test episode
done = False
state = get_state(temp_state=None)
action_lookback = max(args.action_lookback_actor, args.action_lookback_critic)
state_lookback = max(args.state_lookback_actor, args.state_lookback_critic)
# Reset the previous action, as our program factors this into account when taking future actions
if action_lookback > 0:
    prev_actions = np.zeros(action_space_size * action_lookback)
else:
    prev_actions = None
if state_lookback > 0:
    if args.pixel_based:
        prev_states = np.zeros((state_space_size * state_lookback, args.resolution, args.resolution))
    else:
        prev_states = np.zeros(state_space_size * state_lookback)
else:
    prev_states = None

ep_reward = 0
while not done:
        # Sample action from policy, adding noise to state if we want to
        action = agent.select_action(state, prev_states, prev_actions, random_base=args.random_base_train)

        action_noise = np.random.normal(0, 0.1, action_space_size) if args.add_action_noise else 0
        # Take a step in the environment. Note, we get the next state in the following line in case we only want pos
        tmp_st, reward, done, _ = env.step(action + action_noise)  # Step
        next_state = get_state(temp_state=tmp_st)

        # Add state noise if that parameter is true
        next_state += np.random.normal(0, 0.1, state_space_size) if args.add_state_noise else 0
        # Update the cum. reward
        ep_reward += reward

        if action_lookback > 0:
            prev_actions = np.concatenate((prev_actions[action_space_size:], action))
        if state_lookback > 0:
            prev_states = np.concatenate((prev_states[state_space_size:], state))

        state = next_state

print("Test Episode Finised, Reward:", ep_reward)

