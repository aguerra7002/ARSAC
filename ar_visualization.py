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
                   "base_std": [],
                   "adj_shift": [], # TODO: Maybe add later
                   "adj_scale": []}
                   #"state": [],
                   #"action": [],
                   #"reward": []}

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

def make_ar_component_visualization(arsac_visual_dict, steps_to_plot, start_range, end_range):
    # Range of steps to plot in a given episode
    x_axis = np.array(range(start_range, end_range))

    # 3 lines we will will plot
    base_mean_arsac_list = arsac_visual_dict["base_mean"]
    base_std_arsac_list = arsac_visual_dict["base_std"]
    adj_scale_arsac_list = arsac_visual_dict["adj_scale"]
    adj_shift_arsac_list = arsac_visual_dict["adj_shift"]

    # For the eval episodes we want to plot
    for i in range(len(steps_to_plot)):
        # String of eval epsiode
        eval_ep_str = str(int(steps_to_plot[i] / 10000) + 1)

        # Get the data for this eval episode
        base_mean = base_mean_arsac_list[i]
        base_std = base_std_arsac_list[i]
        adj_scale = adj_scale_arsac_list[i]
        adj_shift = adj_shift_arsac_list[i]

        for dim in range(len(base_mean[0])):

            title = "Half-Cheetah, Hidden Dimension 1x32, Eval episode: " + eval_ep_str

            fig, axs = plt.subplots(2)
            fig.suptitle(title)
            # Arsac base mean & std
            mean = np.array([row[dim] for row in base_mean[start_range:end_range]])
            std = np.array([row[dim] for row in base_std[start_range:end_range]])
            axs[0].plot(x_axis, mean)
            axs[0].fill_between(x_axis, mean - std, mean + std, alpha=0.25)
            # Legend
            #axs[0].legend(axs[0].get_lines(), ["SAC", "ARSAC", "ARSAC (base only)"], prop={'size': 10},
            #             title="Action source:", loc="upper left")
            # labels
            axs[0].set_title("Space along Dimension " + str(dim))
            axs[0].set_ylabel("Base Action")
            axs[0].tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)

            # Affine Space Plot
            shift = np.array([row[dim] for row in adj_shift[start_range:end_range]])
            scale = np.array([row[dim] for row in adj_scale[start_range:end_range]])
            axs[1].plot(x_axis, shift)
            axs[1].fill_between(x_axis, shift - scale, shift + scale, alpha=0.25)
            action = mean * scale + shift
            axs[1].plot(x_axis, action, '.', color='black');
            # Legend
            # axs[1].legend(axs[0].get_lines(), ["ARSAC (base only)", "ARSAC", "SAC"], prop={'size': 10},
            #               title="Algorithm:", loc="lower right")
            # Labels
            axs[1].set_ylabel("Post-Affine-Transform Action")
            axs[1].set_xlabel("Step")

            # Save the plot
            fig.savefig("ar_visual/ar_plot_" + eval_ep_str + "_dim_" + str(dim) + '.pdf')
            plt.clf()


if __name__ == "__main__":

    arsac_experiment_id = '3e8aaf6c413a4e09a9e0139a68e6d570'  # Half-Cheetah, al=3, hdb=32, seed 1

    # Get the experiment from comet
    arsac_experiment = comet_api.get_experiment(project_name=project_name,
                                                workspace=workspace,
                                                experiment=arsac_experiment_id)

    # eval episodes to plot (Note: subtract 1 and multiply be 10000  to get step the eval episode occurred at)
    steps_to_plot = [990000]

    # Get the dictionary of the data from Comet
    arsac_visual_dict = get_visual_dict(steps_to_plot, arsac_experiment)

    # Which steps of the episode we will plot
    start_range = 0
    end_range = 50

    # This makes the visualization of the autoregressive component over some range of steps in eval episodes
    make_ar_component_visualization(arsac_visual_dict, steps_to_plot, start_range, end_range)












