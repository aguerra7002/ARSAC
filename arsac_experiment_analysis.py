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
        print(adj_scale.shape, adj_shift.shape, base_mean.shape)
        # Compute the action
        base_mean = np.multiply(base_mean, adj_scale) + adj_shift
    hm = np.zeros((base_mean.shape[1], base_mean.shape[1]))
    for i in range(base_mean.shape[1]):
        vec1 = base_mean[:-offset,i]
        for j in range(base_mean.shape[1]):
            vec2 = base_mean[offset:, j]
            hm[i, j] = np.corrcoef(vec1, vec2)[0,1]

    fig, axs = plt.subplots(1)
    fig.suptitle(title)
    axs.set_xlabel("Actions at t-" + str(offset))
    axs.set_ylabel("Actions")

    axs.imshow((hm+1) / 2, cmap="viridis", interpolation='nearest')
    fig.savefig(save_dir + title.replace(" ", "_") + "_action_corr_offset_" + str(offset) + ".pdf")


if __name__ == "__main__":
    save_dir = "arsac_analysis/"
    # Some walker walk experiment
    arsac_experiment_ids = ['3b7564c7ca044faeaf459e7aa5635b98']
    title = "Walker Walk (HD 32, BS 256)"
    plot_log_scale(arsac_experiment_ids, title, save_dir)
    plot_rewards(arsac_experiment_ids, title, save_dir)
    plot_action_correlation(arsac_experiment_ids[0], title, save_dir)