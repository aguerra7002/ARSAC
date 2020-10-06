from comet_ml.api import API
import numpy as np
from scipy.signal import savgol_filter
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
project_name = 'arsac-test'
comet_api = API(api_key=api_key)

# savgol filter
SMOOTH = True
WINDOW = 71
POLY_DEG = 3

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
    # # Jenky fix for old arsac tests
    # if len(returns) > 100:
    #     returns = returns[1::2]
    #     steps = np.array(range(len(returns))) * 1e3
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
        if len(mean) > WINDOW:
            mean = savgol_filter(mean, WINDOW, POLY_DEG)
        if std is not None and len(std) > WINDOW:
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
            if inference_type == "SAC (2x256)":
                end_mean_array = [mean[-1]] * mean.shape[0]
                plt.plot(end_mean_array, '--', color='black')
            else:
                plot_mean_std(mean, std, inference_type)
        elif len(returns_list) == 1:
            print(len(returns_list))
            plt.plot(savgol_filter(returns_list[0], WINDOW, POLY_DEG), label=inference_type)
    plt.legend(fontsize=12)
    #plt.xlabel(r'Steps $\times 1,000$', fontsize=15)
    plt.xlabel('Eval Episode', fontsize=15)
    plt.ylabel('Cumulative Reward', fontsize=15)
    plt.title(env, fontsize=20)
    name = env.replace("w/", "_").replace(" ", "")
    plt.savefig(save_folder + name + '_rewards.pdf')
    plt.clf()

walker_transfer_dict = {"SAC": ['21ed3d77773e4249b171a321a2bdcd07'],
                 "SAC w/ base transfer": ['2ce5802c9ea845fc9156514c21b53c02'], # Transferred from 0c00a35dea5e425d8af55d3804b3c7b0
                 "ARSAC": ['0e5f9791511242eaa54f175af46a42f7'],
                 "ARSAC w/ flow transfer": ['cf8a735331fc4fc689943cf1581f3c79'], # Transferred from 157216c90f8e400bab5264211ede1646
                 "ARSAC w/ base + flow transfer": ['a131e09fef224a27ab9d5050aa7d2dd5']} # Transferred from 157216c90f8e400bab5264211ede1646

# DM Control Base Tests
walker_walk_base_dict = {"SAC": ['5486ada760c640f7b5fbdd3680ce8258'],
                         "ARSAC-3": ['848a8336f98343dca7e09ff80e221dba']}
walker_run_base_dict = {"SAC": ['a2277a87765e4ddd8d9af641a8ce97cf'],
                         "ARSAC-3": ['f2e735fbf6d0490b8a778371d2af69f3']}
quadruped_walk_base_dict = {"SAC": ['502d3394b8f04710aeeda0522d2765e9'],
                         "ARSAC-3": ['22ece5090c424bf7ba1fd0a00ec57c78']}
quadruped_run_base_dict = {"SAC": ['8c3a67964315417d964ccc86ff2b77f7'],
                         "ARSAC-3": ['88556932cf1341d4bbd76f0bda65a71e']}
hopper_stand_base_dict = {"SAC": ['27553ab85f5d4f2981339ecb244ed92a'],
                         "ARSAC-3": ['6fd32d97a7214344b16da9b6b6ae6d71']}
hopper_hop_base_dict = {"SAC": ['e4c400fa2c654d739952138653271a55'],
                         "ARSAC-3": ['bcdbbdd174764aba84742d2f9122f789']}

# DM Control Pixel Tests
# TODO: FILLIN

to_plot_dict = {
        # "Walker Run Transfer" : walker_transfer_dict,
        "Walker Walk Base" : walker_walk_base_dict,
        "Walker Run Base" : walker_run_base_dict,
        "Quadruped Walk Base" : quadruped_walk_base_dict,
        "Quadruped Run Base" : quadruped_run_base_dict,
        "Hopper Stand Base" : hopper_stand_base_dict,
        "Hopper Hop Base" : hopper_hop_base_dict
    }


if __name__ == "__main__":

    # Specify the folder we want to save the visualizations to
    base_dir = "reward_plots_new/"
    for env in to_plot_dict.keys():
        env_exp_dict = to_plot_dict[env]
        print("Visualizing ", env)

        plot_rewards(env_exp_dict, base_dir)