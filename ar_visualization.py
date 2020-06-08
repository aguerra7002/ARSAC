from comet_ml.api import API
import numpy as np
import matplotlib
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






def get_visual_dict(steps_to_plot, experiment):

    # Get the asset list
    asset_list = experiment.get_asset_list()

    # Empty dict where we will store all the data we wanna visualize
    visual_dict = {"base_mean": [],
                   # "base_std": [],
                   #"adj_shift": [], # TODO: Maybe add later
                   #"adj_scale": [],
                   "state": [],
                   "action": [],
                   "reward": []}

    for step in steps_to_plot:
        # For every data entry we want to log
        for key in visual_dict.keys():
            # Specify the file name
            fileName = "episode_step_" + str(step) + "_" + key

            # Get the asset id
            match = [x for x in asset_list if x['fileName'] == fileName]
            if len(match) == 0:
                continue
            asset_id = [x for x in asset_list if x['fileName'] == fileName][0]['assetId']

            # Now we get the asset
            asset = experiment.get_asset(asset_id, return_type="json")

            # Now simply add it to the visual dict
            visual_dict[key].append(asset)

    return visual_dict

def find_best_start(states1, states2):
    best_norm = 1e9
    best_i = 0

    print(states1[0])
    print("")
    print(states2[0])

    for i in range(len(states1)):
        state1 = np.array(states1[i])
        state2 = np.array(states2[i])
        norm = np.linalg.norm(state1 - state2)
        if norm < best_norm:
            best_norm = norm
            best_i = i

    print(best_i)
    return best_i

def to_cum_rewards(rews):
    return [sum(rews[0:i]) for i in range(len(rews))]


if __name__ == "__main__":

    arsac_experiment_id = '3e8aaf6c413a4e09a9e0139a68e6d570'  # Half-Cheetah, al=3, hdb=32, seed 1
    sac_experiment_id = '8f62350abc7249a2803ad2844b83a3d7' # Half-Cheetah, al=0, hdb=32, seed 1

    # Get the experiment from comet
    arsac_experiment = comet_api.get_experiment(project_name=project_name,
                                                workspace=workspace,
                                                experiment=arsac_experiment_id)
    sac_experiment = comet_api.get_experiment(project_name=project_name,
                                                workspace=workspace,
                                                experiment=sac_experiment_id)

    # eval episodes to plot (Note: subtract 1 and multiply be 10000  to get step the eval episode occurred at)
    steps_to_plot = [990000]

    # Get the dictionary of the data from Comet
    arsac_visual_dict = get_visual_dict(steps_to_plot, arsac_experiment)
    sac_visual_dict = get_visual_dict(steps_to_plot, sac_experiment)

    # For finding the most similar states
    #best_start = find_best_start(arsac_visual_dict["state"][0], sac_visual_dict["state"][0])

    # Range of steps to plot in a given episode
    start_range = 0
    end_range = 50
    x_axis = range(start_range, end_range)

    # 3 lines we will will plot
    arsac_action_list = arsac_visual_dict["action"]
    base_arsac_action_list = arsac_visual_dict["base_mean"]
    sac_action_list = sac_visual_dict["action"]

    # Also will plot the rewards
    arsac_reward_list = arsac_visual_dict["reward"]
    sac_reward_list = sac_visual_dict["reward"]

    # For the eval episodes we want to plot
    for i in range(len(steps_to_plot)):
        # String of eval epsiode
        eval_ep_str = str(int(steps_to_plot[i] / 10000) + 1)

        # Get the data for this eval episode
        arsac_action = arsac_action_list[i]
        base_arsac_action = base_arsac_action_list[i]
        sac_action = sac_action_list[i]

        arsac_reward = arsac_reward_list[i]
        sac_reward = sac_reward_list[i]

        for dim in range(len(arsac_action[0])):

            # Make the plot for this dimension
            title = "Half-Cheetah, Hidden Dimension 32, Eval episode: " + eval_ep_str

            fig, axs = plt.subplots(2)
            fig.suptitle(title)
            # Arsac base action (without AR component)
            axs[0].plot(x_axis, [row[dim] for row in base_arsac_action[start_range:end_range]])
            # Arsac action
            axs[0].plot(x_axis, [row[dim] for row in arsac_action[start_range:end_range]])
            # Sac action
            axs[0].plot(x_axis, [row[dim] for row in sac_action[start_range:end_range]])
            # Legend
            #axs[0].legend(axs[0].get_lines(), ["SAC", "ARSAC", "ARSAC (base only)"], prop={'size': 10},
            #             title="Action source:", loc="upper left")
            # labes
            axs[0].set_title("Action along dimension " + str(dim))
            axs[0].set_ylabel("Action")
            axs[0].tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)

            # Arsac Rewards
            axs[1].plot(x_axis, to_cum_rewards(arsac_reward)[start_range:end_range], color=axs[0].get_lines()[1].get_color())
            # Sac Rewards
            axs[1].plot(x_axis, to_cum_rewards(sac_reward)[start_range:end_range], color=axs[0].get_lines()[2].get_color())
            # Legend
            axs[1].legend(axs[0].get_lines(), ["ARSAC (base only)", "ARSAC", "SAC"], prop={'size': 10},
                          title="Algorithm:", loc="lower right")
            # Labels
            axs[1].set_ylabel("Cumulative rewards")
            axs[1].set_xlabel("Step")

            # Save the plot
            fig.savefig("ar_visual/ar_plot_" + eval_ep_str + "_dim_" + str(dim) + '.png')
            plt.clf()





