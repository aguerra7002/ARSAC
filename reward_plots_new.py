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

# DM Control Transfer tasks
walker_transfer_dict = {"SAC": ['21ed3d77773e4249b171a321a2bdcd07'],
                 "SAC w/ base transfer": ['2ce5802c9ea845fc9156514c21b53c02'], # Transferred from 0c00a35dea5e425d8af55d3804b3c7b0
                 "ARSAC": ['0e5f9791511242eaa54f175af46a42f7'],
                 "ARSAC w/ flow transfer": ['cf8a735331fc4fc689943cf1581f3c79'], # Transferred from 157216c90f8e400bab5264211ede1646
                 "ARSAC w/ base + flow transfer": ['a131e09fef224a27ab9d5050aa7d2dd5']} # Transferred from 157216c90f8e400bab5264211ede1646

# DM Control Base Tests (Batch size 128, ARSAC-3, hidden dim 32)
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

# DM Control Base Tests (Batch Size 256, ARSAC-5, hidden dim 32)
walker_walk_base_dict2 = {"SAC": ['b1acc1d1812b490c91437d8735f37911', # 1
                                  '25cabb2bf78b4a26847cdbf07fee1f73', # 2
                                  '8671749406ad4e4a97372d36e28d1bd0'], # 3
                         "ARSAC-5": ['3b7564c7ca044faeaf459e7aa5635b98',
                                     '14007a1391ca4758a7c07250957902c6',
                                     'c78c27f6b2854a38a9100a688c553b39']}
walker_run_base_dict2 = {"SAC": ['3f271263d96b45009a19b32aa816260b',
                                 'ac28eb5a38b3468b9de3c641ca3f86c2',
                                 '5af66ac575284976a624b25371d99ef6'],
                         "ARSAC-5": ['91cb40a43bb64e23854feb8efe76db8c',
                                     '1441bfaa31954a239f0d3fa0cb789b1a',
                                     '437f1e393c6e4ed5aced57583bda3460']}
quadruped_walk_base_dict2 = {"SAC": ['ea66866a0f4440c4b5c9a979df987ae7',
                                     'a7d9fc48ff7a49fc92faa6d83980ece1',
                                     '475848cab7d24c4cab0bde5c7f4f24d3'],
                         "ARSAC-5": ['68fd82b266614f3db1430cd7c63eaf7e',
                                     '571019e4d63a46b6bab1069b150d3fa9',
                                     '12b06528d71243bbb92516121cdbdf80']}
quadruped_run_base_dict2 = {"SAC": ['af54967243e748cf94d62341712c8336',
                                    '7d869b663a584665b2e7958342a44711',
                                    'c32b0ffc37724d178efa886a1b94cfa6'],
                         "ARSAC-5": ['6f9ba61406c54cefa9cdde52b2766e90',
                                     '8c7010391eea4e0e94a0bcc900b1e550',
                                     '6fe0d6da6c7247d0ae1e2c73959323c6']}
hopper_stand_base_dict2 = {"SAC": ['c2eecc92b7034fab97c712a8d372e9a5',
                                   '6cbfa580e0e349608eaead9286947edb',
                                   'ef9f5a9f20e640d289c21a32c66b6dae'],
                         "ARSAC-5": ['38a6cc65e68f4b9f93261d1a7fa5c13c',
                                     '5ad134f56f92420cb0ea7c308e7cb324',
                                     '9a082948d3cd4325808d1d48e9890c45']}
hopper_hop_base_dict2 = {"SAC": ['84fe85ab190e46829a5ea83ce7f71c66',
                                 '880a4442ab18430bb41d277e1da1d6a5',
                                 '2c09f071cd51428c94af79294e80da1f'],
                         "ARSAC-5": ['94b386e376124a1d8d157ce0a85c758a',
                                     '497aaa6f7cc1481d89339c2a1df0bfd7',
                                     '74404513a2a245c3adcff5b96b1c4254']}

# DM Control Base Tests (Batch Size 256, ARSAC-5, hidden dim 2x256)
# We only do quadruped and hopper since walker was able to learn with 1x32 hidden size
quadruped_walk_base_dict3 = {"SAC": ['829f7aff24e041ef94829e266147965f',
                                     '511fb31a0a6b48fcae11a43c436616f2',
                                     'a73054aa200b42d08978a8487937642d'],
                         "ARSAC-5": ['4d75ad875f5140728c7be89bb0bb96e4',
                                     'cc937d7005274803a640a68c4aa8367c',
                                     'a966793edf034ac79fb63918d410f450']}
quadruped_run_base_dict3 = {"SAC": ['b677035a0e044c5db3a9f598ccc3de90',
                                    '3e0e68b5d2fc4ecdb397a8b2e012bf44',
                                    '1f0f42bbebd8418b81e01c8f21e09d73'],
                         "ARSAC-5": ['d2e43dbd1edc4f2a8074d60f92f8191a',
                                     'a7c6092e814846eb94d0393f896b83b0',
                                     '0aa732f33af84377b61f25a1c9971575']}
hopper_stand_base_dict3 = {"SAC": ['13f01b2093de4e058b5ab71594834d2b',
                                   '9a23165ed514410ca6e8a81f991cd037',
                                   'e6552a8917514a4a888f4d1c3dbe0621'],
                         "ARSAC-5": ['b5cecb25688340f48f691b9cbcf73a6b',
                                     'd75d30f5d83d423ea196c68fa5a93a66',
                                     '5a737d5d786d4c659c76978c3f860ac4']}
hopper_hop_base_dict3 = {"SAC": ['ea594f54e21b4c01921cc2dbfd3f7189',
                                 '09e56e24a85c446fbf76a6c05facceae',
                                 '8a82bace2dbf443fa36051ae08cccd05'],
                         "ARSAC-5": ['b70145be1cdb45a1bdc97b1a849f0d93',
                                     'e73dcb0f4d62404cba985344b31a6a26',
                                     '908fcee3ac8d4156a2e83f8639df8111']}

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

to_plot_dict2 = {
        "Walker Walk Base 256" : walker_walk_base_dict2,
        "Walker Run Base 256" : walker_run_base_dict2,
        "Quadruped Walk Base 256" : quadruped_walk_base_dict2,
        "Quadruped Run Base 256" : quadruped_run_base_dict2,
        "Hopper Stand Base 256" : hopper_stand_base_dict2,
        "Hopper Hop Base 256" : hopper_hop_base_dict2
    }

to_plot_dict3 = {
    "Quadruped Walk Full" : quadruped_walk_base_dict3,
    "Quadruped Run Full" : quadruped_run_base_dict3,
    "Hopper Stand Full" : hopper_stand_base_dict3,
    "Hopper Hop Full" : hopper_hop_base_dict3
}

if __name__ == "__main__":

    # Specify the folder we want to save the visualizations to
    base_dir = "reward_plots_new/"
    for env in to_plot_dict3.keys():
        env_exp_dict = to_plot_dict3[env]
        print("Visualizing ", env)

        plot_rewards(env_exp_dict, base_dir)