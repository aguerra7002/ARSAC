from comet_ml.api import API
import numpy as np
import argparse
import matplotlib
import torch
from arrl import ARRL
import imageio
import os
from env_wrapper import EnvWrapper
import matplotlib.pyplot as plt
from matplotlib import rc
matplotlib.style.use('seaborn')
#rc('text', usetex=True)
rc('font', family='serif')
import seaborn as sns
sns.set_palette('Paired')

# comet
api_key = 'tHDbEydFQGW7F1MWmIKlEvrly'

workspace = 'aguerra'
project_name = 'arsac'
save_dir = "arsac_analysis/"
comet_api = API(api_key=api_key)

# savgol filter
SMOOTH = True
WINDOW = 71
POLY_DEG = 3

def plot_log_scale(experiment_ids, title, save_dir):

    fig, axs = plt.subplots(1)
    fig.suptitle(title)
    axs.set_xlabel("Eval episodes")
    axs.set_ylabel("Avg. Log(Scale)")

    for id in experiment_ids:
        experiment = comet_api.get_experiment(project_name=project_name,
                                              workspace=workspace,
                                              experiment=id)
        asset_list = experiment.get_asset_list()
        for asset in asset_list:
            if asset['fileName'] == 'output_metrics':
                id = asset['assetId']
                visual_dict = experiment.get_asset(id, return_type="json")
                break
        log_scales = []
        for eval in visual_dict:
            log_scale = np.mean(np.log(np.array(eval['adj_scale'])))
            log_scales.append(log_scale)

        axs.plot(log_scales)

    #If we want a legend we use this
    #axs.legend(axs.get_lines(), experiment_dict.keys(), prop={'size': 10}, title="Lookback")
    fig.savefig(save_dir + title.replace(" ", "_") + "_log_scale.pdf")

def run_eval_episode(exp_id, title, plot_agent=False, eval=True, actor_filename='actor.model', override_task=None, prior_only=False, num_steps=1000):
    print("Running eval episode for " + exp_id + " (" + title + ")")
    experiment = comet_api.get_experiment(project_name=project_name,
                                          workspace=workspace,
                                          experiment=exp_id)
    asset_list = experiment.get_asset_list()

    # First setup the arguments
    parser = argparse.ArgumentParser(description='PyTorch AutoRegressiveFlows-RL Args')
    args = parser.parse_args()
    args_asset_id = [x for x in asset_list if x['fileName'] == "args"][0]['assetId']
    args.__dict__ = experiment.get_asset(args_asset_id, return_type="json")

    transfer_domain = args.env_name
    transfer_task = args.task_name if override_task == None else override_task
    env = EnvWrapper(transfer_domain, transfer_task, args.pixel_based,
                     720)  # Make it high res bc we are just visualizing it

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    # Action Space Size
    action_space_size = env.action_space.sample().shape[0]

    # Figure out what the state size is
    state_space_size = env.get_state_space_size(position_only=args.position_only)

    # Now initialize the agent
    agent = ARRL(state_space_size, env.action_space, args)

    # critic_filename = "critic.model"
    actor_asset_id = [x for x in asset_list if actor_filename == x['fileName']][0]['assetId']

    actor = experiment.get_asset(actor_asset_id)
    act_path = '../../tmploaded/actor.model'
    with open(act_path, 'wb+') as f:
        f.write(actor)
    agent.load_model(act_path, None)

    action_lookback = max(args.action_lookback_actor, args.action_lookback_critic)
    state_lookback = max(args.state_lookback_actor, args.state_lookback_critic)

    state = env.get_current_state(temp_state=None, position_only=args.position_only)

    # Reset the previous action, as our program factors this into account when taking future actions
    if action_lookback > 0:
        prev_actions = np.zeros(action_space_size * action_lookback)
    else:
        prev_actions = None
    if state_lookback > 0:
        prev_states = np.zeros(state_space_size * state_lookback)
    else:
        prev_states = None

    with imageio.get_writer('visual/' + title + '.gif', mode='I') as writer:

        img = env.env.physics.render(camera_id=0, width=640, height=480)
        writer.append_data(img)

        actions = np.zeros((action_space_size, num_steps))
        means = np.zeros((action_space_size, num_steps))
        stds = np.zeros((action_space_size, num_steps))
        shifts = np.zeros((action_space_size, num_steps))
        scales = np.zeros((action_space_size, num_steps))
        rewards = np.zeros(num_steps)
        log_probs = np.zeros(num_steps)

        for step in range(num_steps):
            action, mean, std, scale, shift, log_prob = agent.select_action(state, prev_states, prev_actions,
                                                                  random_base=args.random_base_train,
                                                                  eval=eval,
                                                                  return_distribution=True,
                                                                  return_prob = True)
            if args.policy == "Gaussian2":
                log_prob = log_prob[0] # Hacky fix

            # Take a step in the environment. Note, we get the next state in the following line in case we only want pos
            if prior_only and step > 100:
                # Here, when using Gaussian2 policy, mean represents the prior mean which is solely a function of previous actions
                act_prior = np.tanh(mean)
                tmp_st, reward, done, _ = env.step(mean)
            else:
                tmp_st, reward, done, _ = env.step(action)  # Step

            # Determines whether or not to plot the agent moving
            img = env.env.physics.render(camera_id=0, width=640, height=480)
            if plot_agent:
                plt.imshow(img)
                plt.savefig("visual/jpgs/" + title.replace(" ", "_") + "_" + str(step) + ".jpg")
                plt.savefig("visual/pdfs/" + title.replace(" ", "_") + "_" + str(step) + ".pdf")
            writer.append_data(img)

            actions[:, step] = action
            means[:, step] = mean
            stds[:, step] = std
            shifts[:, step] = shift
            scales[:, step] = scale
            rewards[step] = reward
            log_probs[step] = log_prob

            next_state = env.get_current_state(temp_state=tmp_st)

            if action_lookback > 0:
                prev_actions = np.concatenate((prev_actions[action_space_size:], action))
            if state_lookback > 0:
                prev_states = np.concatenate((prev_states[state_space_size:], state))

            state = next_state

    print("Done with the episode, Returning.")
    # Return the relevant info for the entire episode.
    return actions, means, stds, shifts, scales, rewards, log_probs

def get_corr_matrix(base, offset):
    hm = np.zeros((base.shape[0], base.shape[0]))
    for i in range(base.shape[0]):
        if offset == 0:
            vec1 = base[i]
        else:
            vec1 = base[i, :-offset]
        for j in range(base.shape[0]):
            vec2 = base[j, offset:]
            hm[i, j] = np.corrcoef(vec1, vec2)[0, 1]
    return hm

# Plots the correlation statistics among the actions for a given episode.
def plot_action_corr(actions, means, shifts, title, plot_hm=False, actions_sac=None, means_sac=None):
    print("Plotting Correlations")
    action_corr = []
    mean_corr = []
    shift_corr = []
    action_sac_corr = []
    mean_sac_corr = []
    for offset in range(10):
        # Un-tanh the actions to get back into the unscaled realm.
        action_hm = get_corr_matrix(actions, offset)
        mean_hm = get_corr_matrix(means, offset)
        shift_hm = get_corr_matrix(shifts, offset)
        if actions_sac is not None:
            action_sac_hm = get_corr_matrix(actions_sac, offset)
        if means_sac is not None:
            mean_sac_hm = get_corr_matrix(means_sac, offset)
        if plot_hm:
            pass # TODO: Fix the below code? might not be super necessary
            # fig, axs = plt.subplots(1)
            # fig.suptitle(title)
            # axs.set_xlabel("Actions at t-" + str(offset))
            # axs.set_ylabel("Actions")
            #
            # axs.imshow((hm + 1) / 2, cmap="viridis", interpolation='nearest')
            # fig.savefig(save_dir + "action_corr/jpgs/" + title.replace(" ", "_") + "_action_corr_offset_" + str(
            #     offset) + ".jpg")
            # fig.savefig(save_dir + "action_corr/pdfs/" + title.replace(" ", "_") + "_action_corr_offset_" + str(
            #     offset) + ".pdf")
            # plt.clf()

        action_corr.append(np.linalg.norm(action_hm, ord=1))
        mean_corr.append(np.linalg.norm(mean_hm, ord=1))
        shift_corr.append(np.linalg.norm(shift_hm, ord=1))
        if actions_sac is not None:
            action_sac_corr.append(np.linalg.norm(action_sac_hm, ord=1))
        if means_sac is not None:
            mean_sac_corr.append(np.linalg.norm(mean_sac_hm, ord=1))
    plt.plot(action_corr, label="final action")
    plt.plot(mean_corr, label="base dist")
    plt.plot(shift_corr, label="shift comp")
    if actions_sac is not None:
        plt.plot(action_sac_corr, label="SAC final action")
    # Should we plot the mean action?
    # if means_sac is not None:
    #    plt.plot(mean_sac_corr, label="SAC mean")
    plt.legend()
    plt.title(title)
    plt.xlabel("Timesteps")
    plt.ylabel("Norm of correlation matrix")
    plt.savefig("action_corr/jpgs/" + title.replace(" ", "_") + "base_vs_full.jpg")
    plt.savefig("action_corr/pdfs/" + title.replace(" ", "_") + "base_vs_full.pdf")
    plt.clf()

def plot_action(stds, means, shifts, scales, title, start_step=100, end_step=250):
    print("Plotting action")
    # Make an action plot for every dimension
    for dim in range(actions.shape[0]):
        x_axis = range(start_step, end_step)
        fig, axs = plt.subplots(2)
        fig.suptitle(title)
        # Arsac base mean & std
        mean_y = means[dim, start_step:end_step]
        std_y = stds[dim, start_step:end_step]
        axs[0].plot(x_axis, mean_y, color='purple')
        axs[0].fill_between(x_axis, mean_y - std_y, mean_y + std_y, color='purple', alpha=0.25)

        axs[0].set_title("Space along Dimension " + str(dim))
        axs[0].set_ylabel("Prior Action")
        axs[0].tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)

        shift_y = shifts[dim, start_step:end_step]
        scale_y = scales[dim, start_step:end_step]
        axs[1].plot(x_axis, shift_y)
        axs[1].fill_between(x_axis, shift_y - scale_y, shift_y + scale_y, alpha=0.25)
        action_y = shift_y + scale_y * mean_y
        axs[1].plot(x_axis, action_y, '.', color='black')
        axs[1].set_ylabel("Base Policy (blue)/Final Action (black)")
        axs[1].set_xlabel("Step")

        # Save the plot
        plt.savefig("actions/jpgs/" + title.replace(" ", "_") + "_dim" + str(dim) + ".jpg")
        plt.savefig("actions/pdfs/" + title.replace(" ", "_") + "_dim" + str(dim) + ".pdf")
        plt.clf()


def setup_directory(dir):
    loc = os.path.join(save_dir, dir.replace(" ", "_"))
    if not os.path.exists(loc):
        os.makedirs(loc)
        action_corr_loc = os.path.join(loc, "action_corr")
        os.makedirs(action_corr_loc)
        os.makedirs(os.path.join(action_corr_loc, "jpgs"))
        os.makedirs(os.path.join(action_corr_loc, "pdfs"))
        action_loc = os.path.join(loc, "actions")
        os.makedirs(action_loc)
        os.makedirs(os.path.join(action_loc, "jpgs"))
        os.makedirs(os.path.join(action_loc, "pdfs"))
        visual_loc = os.path.join(loc, "visual")
        os.makedirs(visual_loc)
        os.makedirs(os.path.join(visual_loc, "jpgs"))
        os.makedirs(os.path.join(visual_loc, "pdfs"))
    return loc

# Base Tests with 2x256 HS AutoEntropy Tuning, BS 256
walker_walk_base_dict4 = {
    "SAC": ['2ea19479726d44dfa8cd06d94f620178'],
    "ARSAC": ['113982d9ab8b4faf849ef5a2efb712f5']
}
walker_run_base_dict4 = {
    "SAC": ['707acf6bfd714c6582e737db0191743f'],
    "ARSAC": ['cd2b9f003e404a8daff56dea22e0bcb3']
}
quadruped_walk_base_dict4 = {
    "SAC": ['63f05e87479d48869c2f0aac465700ec'],
    "ARSAC": ['a8f7092ff8de48cc97b7e8dd2e18849a']
}
quadruped_run_base_dict4 = {
    "SAC": ['a3116a2d342049258b3b4c078b0900a9'],
    "ARSAC": ['739af68727ff409eaf20e7cca5e9c233']
}
hopper_stand_base_dict4 = {
    "SAC": ['d3649569dbf64475a8e1816261a773c1'],
    "ARSAC": ['e8218e27d9c74d659bc62b9242ab2de2']
}
hopper_hop_base_dict4 = {
    "SAC": ['3945a372be2645c7a6b8efe2951cc3d1'],
    "ARSAC": ['f34a21792fd049af98794b666cf292ff']
}
cheetah_run_base_dict4 = {
    "SAC": ['d4f8852ef3724b1b922a470f1faecc8e'],
    "ARSAC": ['d249049426d04c44bbeb1a817e20369c']
}
humanoid_walk_base_dict4 = {
    #"SAC": ["12bcc10a1a7642248a876e179eb4cd26"],
    "ARSAC": ["12bcc10a1a7642248a876e179eb4cd26"]
}

# Base Tests With 1x32 HS AutoEntropy Tuning, BS 256
walker_walk_base_dict5 = {
    "SAC": ["4a4049ef6cd64db1b9efe21d30f15f40"],
    "ARSAC": ["5691959d6b01421d8dc1b78aaa3937ff"]
}
walker_run_base_dict5 = {
    "SAC" : ["3193d027636e4593a8ba71a84ee21638"],
    "ARSAC" : ["74e14df64e214d8ab68fcdb91bd44cb8"]
}
hopper_stand_base_dict5 = {
    "SAC": ["732e12e250b644d79e02b74871e55daa"],
    "ARSAC": ["bb2b4fc7de334f94819a634b59def873"]
}
hopper_hop_base_dict5 = {
    "SAC": ["b0713cbd76e84f138a8eab35c76e4df0"],
    "ARSAC": ["aeeb746726ef4b07a573a98df0711f9c"]
}
quadruped_walk_base_dict5 = {
    "SAC" : ["f1d1b694ebf74bd1a7afae1d27fe3687"],
    "ARSAC" : ["14ae9513a53a42bf89b65d02f6cdc5e7"]
}
quadruped_run_base_dict5 = {
    "SAC" : ["0eeb188dba754404a41ed786a52d5ac8"],
    "ARSAC" : ["56911a04f19347898de823f159cf0f52"]
}
cheetah_run_base_dict5 = {
    "SAC": ["82891be32c8f4ffb850f4beb1a6c7934"],
    "ARSAC": ["5ab462af89fd4f3ca670838c252d8dc8"]
}

walker_rbo_increase_dict = {
    "ARSAC": ["1ce5cfcacd274f5cbc066c6d0e0073db"] # use "actor_eval_205.model"
}

quadruped_rbo_increase_dict = {
    "ARSAC": ["614c7e95e22c4d4bab5c9b1b39cae13c"] # use "actor_eval_205.model"
}

# Compressing a stand policy for use for humanoid walk
humanoid_stand_rbo_increase_dict = {
    "ARSAC": ["d490b59aeffe44f9ac9435e95f763021"] # use "actor_eval_125.model"
}

# Compressing a walk policy for use for humanoid run
humanoid_walk_rbo_increase_dict = {
    "ARSAC": ["d201596ba5ff4ddba482514e2888ed15"] # use "actor_eval_190.model"
}

humanoid_run_base_dict4 = {
    "ARSAC": ['9979998a4dd3405c8acd7b4ad900ccb4']
}

halfcheetah_gym_dict = {
    "ARSAC": ["0aa921b90c614166818472adc025b451"]
}

# Gated policy with prior
walker_walk_g2 = {
    # "G1 ARSAC": ["5691959d6b01421d8dc1b78aaa3937ff",
    #              "2d26d47b74554c8ca91c7302c616403b",
    #              "f439a152dc6d462596fababa20323359",
    #              "756d6844ce8347819f6f6849893c1825",
    #              "bfee66cd9d6d4a1baced9aecde015bd4"],
    "ARSAC": ["45b36edcd38e4b9b8c0831a27ff6bf4e"]
}

# Sort of a reverse of g1
walker_walk_g3 = {
    "ARSAC": ["e3446c4a9f9c4ee5a1ad1713f462fb7e"]
}

to_plot_dict_1x32 = {
    #"Walker Walk AutoEnt 1x32HS": walker_walk_base_dict5,
    #"Walker Run AutoEnt 1x32 HS": walker_run_base_dict5,
    # "Hopper Stand AutoEnt 1x32 HS": hopper_stand_base_dict5,
    # "Hopper Hop AutoEnt 1x32 HS": hopper_hop_base_dict5,
    # "Quadruped Walk AutoEnt 1x32 HS": quadruped_walk_base_dict5,
    # "Quadruped Run AutoEnt 1x32 HS": quadruped_run_base_dict5,
    #"Cheetah Run AutoEnt 1x32 HS": cheetah_run_base_dict5
    # "Walker Walk RBO AutoEnt 1x32 HS": walker_rbo_increase_dict,
    # "Quadruped Walk RBO AutoEnt 1x32 HS": quadruped_rbo_increase_dict
    # "Gym HalfCheetah 1x32 HS": halfcheetah_gym_dict
    # "Walker Walk G2 1x32HS": walker_walk_g2,
    "Walker Walk G3 1x32HS": walker_walk_g3
}

to_plot_dict_2x256 = {
    # "Walker Walk AutoEnt 2x256HS": walker_walk_base_dict4,
    # "Walker Run AutoEnt 2x256HS": walker_run_base_dict4,
    # "Hopper Stand AutoEnt 2x256HS": hopper_stand_base_dict4,
    # "Hopper Hop AutoEnt 2x256HS": hopper_hop_base_dict4,
    # "Quadruped Walk AutoEnt 2x256HS": quadruped_walk_base_dict4,
    # "Quadruped Run AutoEnt 2x256HS": quadruped_run_base_dict4,
    # "Cheetah Run AutoEnt 2x256HS": cheetah_run_base_dict4,
    # "Humanoid Walk AutoEnt 2x256HS": humanoid_walk_base_dict4
    # "Humanoid Stand RBO AutoEnt 2x256HS": humanoid_stand_rbo_increase_dict,
    #"Humanoid Walk RBO AutoEnt 2x256HS": humanoid_walk_rbo_increase_dict
    "Humanoid Run AutoEnt 2x256HS": humanoid_run_base_dict4
}

if __name__ == "__main__":

    for key in to_plot_dict_1x32:
        dir_name = key
        # Will create all the necessary directories and go into the proper directory for plotting
        os.chdir(setup_directory(dir_name))
        title = key
        arsac_exp_id = to_plot_dict_1x32[key]["ARSAC"][0]
        actions, means, stds, shifts, scales, rewards, log_probs = run_eval_episode(arsac_exp_id, title, actor_filename="actor.model", prior_only=False)

        #sac_exp_id = to_plot_dict_1x32[key]["SAC"][0]
        #actions_sac, means_sac, stds_sac, _, _, rewards_sac, log_probs_sac = run_eval_episode(sac_exp_id, key, eval=False)
        #print("ARSAC Log Prob", np.mean(log_probs), "SAC Log Prob", np.mean(log_probs_sac))


        print("Episode Reward", sum(rewards))
        plot_action(stds, means, shifts, scales, title)
        # amax = np.max(actions)
        # amin = np.min(actions)
        # if amax >= 1 or amin <= -1:
        #     to_use = np.arctanh(0.99999 * (2*actions - (amax + amin)) / (amax - amin))
        # else:
        #     to_use = np.arctanh(actions)
        # amax = np.max(actions_sac)
        # amin = np.min(actions_sac)
        # if amax >= 1 or amin <= -1:
        #     to_use2 = np.arctanh(0.99999 * (2 * actions_sac - (amax + amin)) / (amax - amin))
        # else:
        #     to_use2 = np.arctanh(actions_sac)
        # plot_action_corr(to_use, means, shifts, title, actions_sac=to_use2, means_sac=means_sac)
        # Reset to original directory (go up two levels, ALWAYS)
        os.chdir("../../")
        print("Done.")