from comet_ml.api import API
import numpy as np
import argparse
import matplotlib
import torch
from arrl import ARRL
import imageio
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

def plot_rewards(experiment_ids, title, save_dir):
    fig, axs = plt.subplots(1)
    fig.suptitle(title)
    axs.set_xlabel("Eval episodes")
    axs.set_ylabel("Cumulative Reward")

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
        rewards = []
        for eval in visual_dict:
            rewards.append(sum(eval["reward"]))
        axs.plot(rewards)

    # If we want a legend we use this
    # axs.legend(axs.get_lines(), experiment_dict.keys(), prop={'size': 10}, title="Lookback")
    fig.savefig(save_dir + title.replace(" ", "_") + "_rewards.pdf")

def plot_action_correlation(id, title, save_dir, eval_to_plot=0, offset=1, base_dist=False):
    # TODO: Somehow extend this for multiple experiments
    # for id in experiment_ids:
    experiment = comet_api.get_experiment(project_name=project_name,
                                          workspace=workspace,
                                          experiment=id)
    asset_list = experiment.get_asset_list()
    for asset in asset_list:
        if asset['fileName'] == 'output_metrics':
            id = asset['assetId']
            visual_dict = experiment.get_asset(id, return_type="json")
            break
    base_mean = np.array(visual_dict[eval_to_plot]["base_mean"])
    if not base_dist:
        adj_scale = np.array(visual_dict[eval_to_plot]["adj_scale"])
        adj_shift = np.array(visual_dict[eval_to_plot]["adj_shift"])
        # Compute the action
        base_mean = np.multiply(base_mean, adj_scale) + adj_shift
    hm = np.zeros((base_mean.shape[1], base_mean.shape[1]))
    for i in range(base_mean.shape[1]):
        if offset == 0:
            vec1 = base_mean[:, i]
        else:
            vec1 = base_mean[:-offset,i]
        for j in range(base_mean.shape[1]):
            vec2 = base_mean[offset:, j]
            hm[i, j] = np.corrcoef(vec1, vec2)[0,1]

    fig, axs = plt.subplots(1)
    fig.suptitle(title)
    axs.set_xlabel("Actions at t-" + str(offset))
    axs.set_ylabel("Actions")

    axs.imshow((hm+1) / 2, cmap="viridis", interpolation='nearest')
    fig.savefig(save_dir + "action_corr/jpgs/" + title.replace(" ", "_") + "_action_corr_offset_" + str(offset) + ".jpg")
    fig.savefig(save_dir + "action_corr/pdfs/" + title.replace(" ", "_") + "_action_corr_offset_" + str(offset) + ".pdf")
    plt.clf()
    return np.linalg.norm(hm) / np.size(hm)

def plot_action_correlation_range(id, title, save_dir, offsets):
    cor_bases = []
    cor_acts = []
    for t in offsets:
        print(t)
        cor_bases.append(plot_action_correlation(id, title + " Base Only", save_dir, offset=t, base_dist=True))
        cor_acts.append(plot_action_correlation(id, title, save_dir, offset=t))
    plt.plot(cor_acts, label="AR Action")
    plt.plot(cor_bases, label="Base Action")
    plt.legend()
    plt.savefig(save_dir + "action_corr/jpgs/" + title.replace(" ", "_") + "base_vs_full.jpg")
    plt.savefig(save_dir + "action_corr/pdfs/" + title.replace(" ", "_") + "base_vs_full.pdf")
    plt.clf()

def plot_eval_episode(exp_id, title, min_steps=70, num_steps=80, plot_action=True, plot_agent=True):
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
    transfer_task = args.task_name
    env = EnvWrapper(transfer_domain, transfer_task, args.pixel_based, 720) # Make it high res bc we are just visualizing it

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    # Action Space Size
    action_space_size = env.action_space.sample().shape[0]

    # Figure out what the state size is
    state_space_size = env.get_state_space_size(position_only=args.position_only)

    # Now initialize the agent
    agent = ARRL(state_space_size, env.action_space, args)

    actor_filename = "actor.model"
    # critic_filename = "critic.model"
    actor_asset_id = [x for x in asset_list if actor_filename == x['fileName']][0]['assetId']

    actor = experiment.get_asset(actor_asset_id)
    act_path = 'tmploaded/actor.model'
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

    with imageio.get_writer('arsac_analysis/visual/gifs/' + title+ '.gif', mode='I') as writer:

        img = env.env.physics.render(camera_id=0, width=640, height=480)
        writer.append_data(img)

        means = np.zeros((num_steps - min_steps, action_space_size))
        shifts = np.zeros((num_steps - min_steps, action_space_size))
        scales = np.zeros((num_steps - min_steps, action_space_size))

        for step in range(num_steps):
            if step % 10 == 0:
                print(step)
            action, mean, std, scale, shift = agent.select_action(state, prev_states, prev_actions,
                                                                  random_base=args.random_base_train,
                                                                  eval=True,
                                                                  return_distribution=True)
            # Take a step in the environment. Note, we get the next state in the following line in case we only want pos
            tmp_st, reward, done, _ = env.step(action)  # Step

            # Determines whether or not to plot the agent moving
            if plot_agent:
                img = env.env.physics.render(camera_id=0, width=640, height=480)
                if step >= min_steps:
                    plt.imshow(img)
                    plt.savefig("arsac_analysis/visual/jpgs/" + title.replace(" ", "_") + "_" + str(step) + ".jpg")
                    plt.savefig("arsac_analysis/visual/pdfs/" + title.replace(" ", "_") + "_" + str(step) + ".pdf")
                writer.append_data(img)
            if plot_action:
                if step >= min_steps:
                    means[step - min_steps] = mean
                    shifts[step - min_steps] = shift
                    scales[step - min_steps] = scale

            next_state = env.get_current_state(temp_state=tmp_st)

            if action_lookback > 0:
                prev_actions = np.concatenate((prev_actions[action_space_size:], action))
            if state_lookback > 0:
                prev_states = np.concatenate((prev_states[state_space_size:], state))

            state = next_state

        print("Plotting action now")
        if plot_action:
            # Now we plot the actions in a bunch of different ways
            # First we do a plot of all the ar components on one dimension
            for dim in range(action_space_size):
                shifts_col = shifts[:,dim]
                plt.plot(shifts_col, label="dim " + str(dim))
            plt.savefig("arsac_analysis/actions/jpgs/" + title.replace(" ", "_") + "_ar_comp.jpg")
            plt.savefig("arsac_analysis/actions/pdfs/" + title.replace(" ", "_") + "_ar_comp.pdf")
            plt.clf()
            # Then we plot the actions
            for dim in range(action_space_size):
                actions_col = means[:,dim] * scales[:, dim] + shifts[:, dim]
                plt.plot(actions_col, label="dim " + str(dim))
            plt.savefig("arsac_analysis/actions/jpgs/"+ title.replace(" ", "_")  + "_actions.jpg")
            plt.savefig("arsac_analysis/actions/pdfs/" + title.replace(" ", "_") + "_actions.pdf")
            plt.clf()
            # Then we plot each dimension (similar to what we had in the workshop paper)
            for dim in range(action_space_size):

                shifts_col = shifts[:, dim]
                scales_col = scales[:, dim]
                actions_col = means[:,dim] * scales[:, dim] + shifts[:, dim]
                x_axis = np.arange(shifts_col.shape[0])
                plt.title("Action and AR Component of dimension " + str(dim))
                plt.plot(x_axis, means[:, dim])
                plt.plot(x_axis, shifts_col)
                plt.fill_between(x_axis, shifts_col - scales_col, shifts_col + scales_col, alpha=0.25)
                plt.plot(x_axis, actions_col, '.', color='black')
                plt.savefig("arsac_analysis/actions/jpgs/" + title.replace(" ", "_") + "_dim" + str(dim) + ".jpg")
                plt.savefig("arsac_analysis/actions/pdfs/" + title.replace(" ", "_") + "_dim" + str(dim) + ".pdf")
                plt.clf()

if __name__ == "__main__":
    save_dir = "arsac_analysis/"
    # Walker Walk (HD 1x32, BS 256, No AutoEnt Tuning)
    walker_walk_experiment1 = ["Walker Walk HD 1x32, BS 256", '3b7564c7ca044faeaf459e7aa5635b98']
    walker_walk_experiment2 = ["Walker Walk AutoEnt HD 2x256, BS 256", '0db2b5f019f3424eb0c17a34265932e3']
    walker_run_experiment1 = ["Transfer Walker Run AutoEnt HD 2x256, BS 256", 'db603b5dabae46eea227c2232d4ec3a5']
    walker_run_experiment2 = ["Walker Run AutoEnt HD 1x32, BS256", "74e14df64e214d8ab68fcdb91bd44cb8"]
    quadruped_walk_experiment = ["Quadruped Walk AutoEnt HD 1x32, BS 256", "14ae9513a53a42bf89b65d02f6cdc5e7"]

    plot_action_correlation_range(walker_run_experiment2[1], walker_run_experiment2[0], save_dir, range(10))

    # Will plot the agent moving for given range of steps
    plot_eval_episode(walker_run_experiment2[1], walker_run_experiment2[0], num_steps = 200, min_steps=70, plot_agent=False)
    #plot_eval_episode(quadruped_walk_experiment[1], quadruped_walk_experiment[0], num_steps=200, min_steps=70, plot_agent=False)