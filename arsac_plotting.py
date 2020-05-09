import json
from os.path import isfile
from pathlib import Path

from comet_ml.api import API
import numpy as np
from celluloid import Camera
from scipy.signal import savgol_filter
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
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


def get_visual_dict_comet(experiment_id, save_json_dir=None):
    # Get the experiment from comet
    experiment = comet_api.get_experiment(project_name=project_name,
                                              workspace=workspace,
                                              experiment=experiment_id)

    # Get the asset list
    asset_list = experiment.get_asset_list()

    # Empty dict where we will store all the data we wanna visualize
    visual_dict = {"base_mean": [],
                   "base_std": [],
                   "adj_shift": [],
                   "adj_scale": []}

    # For each step where we do an eval episode
    steps = [str(i * 5000) for i in range(2, 200)]
    for step in steps:
        #print("Retrieving Data from step:", step)
        # For every data entry we want to log
        for key in visual_dict.keys():
            # Specify the file name
            fileName = "episode_step_" + step + "_" + key

            # Get the asset id
            match = [x for x in asset_list if x['fileName'] == fileName]
            if len(match) == 0:
                continue
            asset_id = [x for x in asset_list if x['fileName'] == fileName][0]['assetId']

            # Now we get the asset
            asset = experiment.get_asset(asset_id, return_type="json")

            # Now simply add it to the visual dict
            visual_dict[key].append(asset)

    # Temporary additions
    with open(save_json_dir + experiment_id + ".json", 'w') as f:
        json.dump(visual_dict, f)

    return visual_dict


def get_returns(experiment):
    """
    Obtains the (eval) returns from an experiment.

    Args:
        experiment (Experiment): a comet experiment object

    Returns a numpy array [n_episodes,] of cumulative rewards (returns).
    """
    # asset_list = experiment.get_asset_list()
    # returns_asset_list = [a for a in asset_list if '_reward' in a['fileName']]
    # if len(returns_asset_list) > 0:
    #     returns = []
    #     steps = []
    #     for asset in returns_asset_list:
    #         data = experiment.get_asset(asset['assetId'], return_type='json')
    #         returns.append(sum(data))
    #         steps.append(float(asset['step']))
    #     steps = np.array(steps)
    #     print(steps[-5:])
    # else:
    returns_asset_list = experiment.get_metrics('train_Avg. Episode_Reward')
    returns = [float(a['metricValue']) for a in returns_asset_list]
    steps = np.array([float(a['step']) for a in returns_asset_list])
    return np.array(returns), steps / 1e3


def aggregate_returns(returns_list):
    """
    Aggregates (mean and std.) the returns from multiple experiments.

    Args:
        returns_list (list): a list of numpy arrays containing returns

    Returns numpy arrays of mean and std. dev of returns.
    """
    if len(returns_list) == 1:
        return returns_list[0], None
    min_episodes = min([exp.shape[0] for exp in returns_list])
    returns_array = np.stack([r[:min_episodes] for r in returns_list])
    returns_mean = returns_array.mean(axis=0)
    returns_std = returns_array.std(axis=0)
    return returns_mean, returns_std


def plot_mean_std(mean, std, label):
    if SMOOTH:
        if len(mean) > 0:
            mean = savgol_filter(mean, WINDOW, POLY_DEG)
        if std is not None and len(std) > 0:
            std = savgol_filter(std, WINDOW, POLY_DEG)
    plt.plot(mean, label=label)
    if std is not None:
        plt.fill_between(np.arange(mean.shape[0]), mean - std, mean + std, alpha=0.2)


def plot_rewards(env_exp_dict, save_folder):
    for inference_type in env_exp_dict:
        returns_list = []
        for exp_key in env_exp_dict[inference_type]:
            if exp_key == '':
                continue
            experiment = comet_api.get_experiment(project_name=project_name,
                                              workspace=workspace,
                                              experiment=exp_key)
            if experiment is not None:
                returns, steps = get_returns(experiment)
                returns_list.append(returns)
            else:
                print("Invalid Comet ML Experiment Key provided: " + exp_key)
        if len(returns_list) > 1:
            mean, std = aggregate_returns(returns_list)
            plot_mean_std(mean, std, inference_type)
        elif len(returns_list) == 1:
            plt.plot(steps, savgol_filter(returns_list[0], WINDOW, POLY_DEG), label=inference_type)
    plt.legend(fontsize=12)
    #plt.xlabel(r'Steps $\times 1,000$', fontsize=15)
    plt.xlabel('Steps x1000', fontsize=15)
    plt.ylabel('Cumulative Reward', fontsize=15)
    plt.title(env, fontsize=20)
    plt.savefig(save_folder + 'rewards.png')
    plt.clf()


def plot_density_animation(visual_dict, save_folder):
    # Set up the figure
    ax_cols = 2
    ax_rows = int(len(visual_dict.keys()) / ax_cols) # Assumes an even number of keys in visual dict
    fig = plt.figure(figsize=(10, 7), dpi=100)
    ax = []
    for j in range(len(visual_dict.keys())):
        ax.append(fig.add_subplot(ax_rows, ax_cols, j+1))

    camera = Camera(fig)

    colors = ["blue", "orange", "green", "red", "purple", "brown", "deeppink", "darkgreen", "lime"]

    # Plot formatting (done manually for this Half Cheetah)
    # ax[0].set_xlim(-6,6)
    # ax[1].set_xlim(0, 4)
    # ax[2].set_xlim(-1, 1)
    # ax[3].set_xlim(0, 1.5)

    len_dict = len(visual_dict["base_mean"])

    for eval_ep_num in range(1, len_dict, 5):
        for j, visual_key in enumerate(visual_dict.keys()):
            # Get the data
            eval_ep = visual_dict[visual_key][eval_ep_num]
            #print(len(eval_ep), len(eval_ep[0]))
            eval_ep_t = list(zip(*eval_ep))
            for i in range(len(eval_ep_t)):
                # Make sure all the elements of the list are not all the same
                if len(set(eval_ep_t)) > 1:
                    # Draw the density plot
                    ax[j] = sns.distplot(eval_ep_t[i], hist=False, kde=True,
                                      kde_kws={'linewidth': 3},
                                      color=colors[i], ax=ax[j])
                    # Alternatively we can draw the line plot as the episode plays out
                    #ax[j].plot(eval_ep_t[i], color=colors[i])


            # Update the frame title
            ax[j].text(0.4, 1.01, visual_key, transform=ax[j].transAxes)

        # Only want 1 legend to avoid redundancy
        ax[1].legend(ax[1].get_lines(), [str(i) for i in range(len(eval_ep_t))], prop={'size': 10}, title="Output Dim")
        # TODO: Make this print relative to the frame
        # ax[0].text(5, 0.86, "Eval Epsiode " + str(eval_ep_num), fontsize=12)
        # Grab the frame
        camera.snap()

    anim = camera.animate()
    anim.save(save_folder + 'density_animation.gif', dpi=100, writer=PillowWriter())
    # After we have saved this, we wanna clear the figure
    plt.clf()

######### API KEYS ##########
exp_dict = {'Hopper-v2': {'Action Lookback = 0': ['bc98b4ca718d4a20b69ffe8ed1c777b4',
                                                  '79c71897f0204402bfc99fcf81d45cc6',
                                                  '00332bf9f72f4d57bb63fc5a76c902c7',
                                                  '2d48f44fab344aefb37ccf610c037d93',
                                                  'aeb80ba9fda44a8fad989017ee673149',
                                                  '000b0603e59b4eba9a2c2d89ec7413c3'],
                          'Action Lookback = 1': ['f7e4e156f5ee495985f77d15d62365e8',
                                                  '2d78243f64974c888904609fc8a04198',
                                                  '58a84fa3fb1b4112b7302f8f6759cd43'],
                          'Action Lookback = 3': ['0bfd50fbd267460fad93d5f84607d35d',
                                                  '2079085ac7e34ab99d429f2be1174cdb',
                                                  'a89026b4717046eea379fb7abe32c2f5']},
            'Walker2d-v2': {'Action Lookback = 0': ['c4a12a88d7674da2a6c8cafa0310f24c',
                                                    'a426ae2bd095474e9316cdad581086c0',
                                                    'd4ca797839a5412297288f67a1f14b85',
                                                    '83fe80214c9c4ee68babc2e7705a654a',
                                                    '0c5ac0bc0a214a599ed4a866e81829d1',
                                                    '1236a09e06ee4e61b919907ec2d5a3cd'],
                            'Action Lookback = 1': ['d99c0da3d97942a4b7f94f06f0f2bc9b',
                                                    '2beea84556b3401da4da65675fd8e97d',
                                                    'f68c47652b364650ad01b93456ed116c'],
                            'Action Lookback = 3': ['3bcc58d1e2644e5ebc8c847087064e4b',
                                                    '91b1865bcbeb43dbb2f7730363e00ab8',
                                                    'e557cf1ffa0d41a9a910606853ae2957']},
            'HalfCheetah-v2': {'Action Lookback = 0': ['2ca07c0a03fe467c810caa7851b9e7cc',
                                                       '9afdc7eff0ca48d4b62b8d00ee6b1090',
                                                       '87407b11b3734f61bcff097568db1fa5',
                                                       '0f98107d6fcb4607b9d0e6e48cb648f6',
                                                       'f9bf149914d34730b35be2c0171d010b',
                                                       '327149f8f2ec456b8d58e27dca9e48a7'],
                               'Action Lookback = 1': ['67d94f574c5647c9952e2df10a584e47',
                                                       'e6149aedc01d4666be36180f3dd81cf8',
                                                       '83abd7e4a63f48758df01541faaa6f1b'],
                               'Action Lookback = 3': ['5ab8c92e50614dc898d4e3a5f1f46604',
                                                       '5ca1185424a3442ebf8b6ace07244c8c',
                                                       '0c5e41aee72944f294760aab96c4b99c']},
            'Ant-v2': {'Action Lookback = 0': ['bd7e539e1a5c4809b2ad4f41d487dfc0',
                                                       'fb71d7453ecb41beada21b934e90ce9e',
                                                       '91e5f5481e3e4e07aec0b9806ced7816'],
                               'Action Lookback = 1': ['99e30b7e9ed44e73a19ffc8e1e2c087a',
                                                       '4873f25ec8ba4e7a9753a8cead77c270',
                                                       'e18299ec599047409445a1e6e3502bda'],
                               'Action Lookback = 3': ['7bc03d8b6c5c445a9fb973e76eea5566',
                                                       '2e48164404a54bda9f7b5adf60baaa44',
                                                       '14cd170d78da415ab7590ebc5e494d00']},
            'Humanoid-v2': {'Action Lookback = 0': ['',
                                                       '',
                                                       ''],
                               'Action Lookback = 1': ['',
                                                       '',
                                                       ''],
                               'Action Lookback = 3': ['',
                                                       '',
                                                       '']}}
##################################



if __name__ == "__main__":

    # Specify the folder we want to save the visualizations to
    base_dir = "visualizations/"
    for env in exp_dict.keys():
        env_exp_dict = exp_dict[env]
        print("Visualizing ", env)
        ######### Plotting the distributions ##########
        #
        # for inference_type in env_exp_dict:
        #     print("  -", inference_type)
        #     # For this, we will only plot a single experiment run to see how training proceeded.
        #     experiment_id = env_exp_dict[inference_type][0]
        #     if experiment_id == '':
        #         continue
        #     lb = inference_type[-1]
        #     save_dir = base_dir + env + "/arsac" + lb + "/"  # Plain ARSAC algorithm, specified by lookback
        #
        #     # Create the directory if it does not already exist
        #     Path(save_dir).mkdir(parents=True, exist_ok=True)
        #
        #     if isfile(save_dir + experiment_id + ".json"):
        #         with open(save_dir + experiment_id + ".json") as f:
        #             visual_dict = json.load(f)
        #     else:
        #         visual_dict = get_visual_dict_comet(experiment_id, save_json_dir=save_dir)
        #
        #     # Now we make the plots
        #     plot_density_animation(visual_dict, save_dir)

        #########    Plotting the rewards    ##########

        plot_rewards(env_exp_dict, base_dir + env + "/")

