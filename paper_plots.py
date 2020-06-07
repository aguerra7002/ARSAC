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
project_name = 'arsac'
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
    plt.savefig(save_folder + env.strip() + '_rewards.png')
    plt.clf()

# Normal SAC Dictionaries to plot
# Hopper with normal SAC
hopper_sac_dict = {"Hidden_Dim 1x32" : ['58f9d24c018840568715e42fe99a158b',
                                        '7337b870ba4145f2947c4a040bcca660',
                                        '3c66858d17b84c58bb030323f2fb0304'],
                   "Hidden Dim 1x128" : ['013a1b9c12ce46f2bf1fa748d4bc8f09',
                                         'dcb2b248c1f147ae9c8f523797e2f026',
                                         '66cd3eea5f7c423d8af6addcc6fca678'],
                   "Hidden Dim 2x256" : ['000b0603e59b4eba9a2c2d89ec7413c3',
                                         'aeb80ba9fda44a8fad989017ee673149',
                                         '2d48f44fab344aefb37ccf610c037d93']}
# Half Cheetah with normal SAC
halfcheetah_sac_dict = {"Hidden_Dim 1x32" : ['f2586bb8002049b0838cae82d55d56c5',
                                             '8f62350abc7249a2803ad2844b83a3d7',
                                             '56dbe81114e5422786c074c7f97a6fff'],
                        "Hidden Dim 1x128" : ['53ff66634650478aa4f69838e2059810',
                                              'e279da0b42b84962baf314653a9b866e',
                                              '8277dfc87bd24a519ac43a6e6f01a974'],
                        "Hidden Dim 2x256" : ['0f98107d6fcb4607b9d0e6e48cb648f6', # Might need to rerun
                                              'f9bf149914d34730b35be2c0171d010b',
                                              '327149f8f2ec456b8d58e27dca9e48a7']}
# Walker with normal SAC
walker_sac_dict = {"Hidden_Dim 1x32" : ['0e29d4f0b9494009ac79b61197951d3e',
                                        '28b8002f78ae477fac617820d80e1a0f',
                                        'a65fd1dc7aad4d24b2badfd35d923abe'],
                   "Hidden Dim 1x128" : ['334a2c1ac59b4e57800b9dfe4e300061',
                                         '9f6e9c51995243ceadc37c3994988c31',
                                         'c9c2b1c094db4651add96076decfeb4d'],
                   "Hidden Dim 2x256" : ['d4ca797839a5412297288f67a1f14b85',
                                         'a426ae2bd095474e9316cdad581086c0',
                                         'c4a12a88d7674da2a6c8cafa0310f24c']}
# Ant with normal SAC
ant_sac_dict = {"Hidden_Dim 1x32" : ['27e9eb4c2caa4830a7485ed7fbe5c3c0',
                                     '7266a1ddb85c4cab9ac5f04b56d74fb4',
                                     '0d45d38335684e86b2f1b09894ab23e3'],
                "Hidden Dim 1x128" : ['6d9c925bc23c4953b432b3108fab8005',
                                      '9d7ced7642874b07854b8d83833ab246',
                                      'a3db6861354e4456a9f5cad99773eee0'],
                "Hidden Dim 2x256" : ['',
                                      '', # Need to be run for 3m steps
                                      '']}

# Normal SAC Dictionaries to plot
# Hopper with autoregressive SAC
hopper_arsac_dict = {"Hidden_Dim 1x32" : ['2a202a1eb9a64a4daffedd134985d190',
                                          '3b5929c73bcc48c1bd15181dcd931abd',
                                          'a2d7b537ce3f464292eabdf0ed220d5d'],
                     "Hidden Dim 1x128" : ['4dffe2e6c3204260893c8eb0b83e02a7',
                                           'c67aacf2a30f49a8a5b2baa155355855',
                                           '1489d7f3242d49e59892addbdc4c994a'],
                     "Hidden Dim 2x256" : ['0bfd50fbd267460fad93d5f84607d35d',
                                           '2079085ac7e34ab99d429f2be1174cdb',
                                           'a89026b4717046eea379fb7abe32c2f5']}
# Half Cheetah with autoregressive SAC
halfcheetah_arsac_dict = {"Hidden_Dim 1x32" : ['3e8aaf6c413a4e09a9e0139a68e6d570',
                                               'a43d37bd14884bff9de4a942f34111d2',
                                               '6ed2b5ebc25b44f4be341306bfed0e44'],
                          "Hidden Dim 1x128" : ['7642f7fb6edb49b3bf20a4969462ed97',
                                                '1e725b1b8cf24632a9362bb68d47b6c3',
                                                'c041e3e3611442ccb7dff13394562b83'],
                          "Hidden Dim 2x256" : ['5ab8c92e50614dc898d4e3a5f1f46604',
                                                '5ca1185424a3442ebf8b6ace07244c8c',
                                                '0c5e41aee72944f294760aab96c4b99c']}
# Walker with autoregressive SAC
walker_arsac_dict = {"Hidden_Dim 1x32" : ['3db6835ad7d84ddfb7222ac3a7aba6d9',
                                          'fe689529bc8d4f7daf5deefeeda7ad2d',
                                          '76a9bd7321bf492cbbf9b556c33d4038'],
                     "Hidden Dim 1x128" : ['',
                                           '',
                                           ''],
                     "Hidden Dim 2x256" : ['',
                                           '',
                                           '']}
# Ant with autoregressive SAC
ant_arsac_dict = {"Hidden_Dim 1x32" : ['',
                                       '',
                                       ''],
                  "Hidden Dim 1x128" : ['',
                                        '',
                                        ''],
                  "Hidden Dim 2x256" : ['',
                                        '',
                                        '']}

to_plot_dict = {
        "Hopper-v2 w/ SAC" : hopper_sac_dict,
        "Hopper-v2 w/ ARSAC" : hopper_arsac_dict,
        "HalfCheetah-v2 w/ SAC" : halfcheetah_sac_dict,
        "HalfCheetah-v2 w/ ARSAC" : halfcheetah_arsac_dict,
        "Walker2d-v2 w/ SAC" : walker_sac_dict,
        "Walker2d-v2 w/ ARSAC" : walker_arsac_dict,
        "Ant-v2 w/ SAC" : ant_sac_dict,
        "Ant-v2 w/ ARSAC" : ant_arsac_dict,
    }

if __name__ == "__main__":

    # Specify the folder we want to save the visualizations to
    base_dir = "paper_plots/"
    for env in to_plot_dict.keys():
        env_exp_dict = to_plot_dict[env]
        print("Visualizing ", env)

        plot_rewards(env_exp_dict, base_dir)