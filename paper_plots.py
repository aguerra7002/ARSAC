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
WINDOW = 37#71
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
    # Jenky fix for old arsac tests
    if len(returns) > 100:
        returns = returns[1::2]
        steps = np.array(range(len(returns))) * 1e3
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
            print(len(mean), WINDOW)
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
            plot_mean_std(mean, std, inference_type)
        elif len(returns_list) == 1:
            plt.plot(steps, savgol_filter(returns_list[0], WINDOW, POLY_DEG), label=inference_type)
    plt.legend(fontsize=12)
    #plt.xlabel(r'Steps $\times 1,000$', fontsize=15)
    plt.xlabel('Steps x1000', fontsize=15)
    plt.ylabel('Cumulative Reward', fontsize=15)
    plt.title(env, fontsize=20)
    name = env.replace("w/", "_").replace(" ", "")
    plt.savefig(save_folder + name + '_rewards.png')
    plt.clf()

# Normal SAC Dictionaries to plot
# Hopper with normal SAC
hopper_sac_dict = {"Hidden_Dim 1x32" : ['f675f9f430404111a9530dfb4d276273', # 4
                                        'ca092a2dd00e48008d43e068aec3ad72', # 5
                                        '0223b49a689b4ab190134ca519eaceb6', # 6
                                        '8efbdefa14da4cc387c7dc9649c57943'], # 7
                   "Hidden Dim 1x128" : ['013a1b9c12ce46f2bf1fa748d4bc8f09', # 0
                                         'dcb2b248c1f147ae9c8f523797e2f026', # 1
                                         '66cd3eea5f7c423d8af6addcc6fca678'], # 2
                   "Hidden Dim 2x256" : ['000b0603e59b4eba9a2c2d89ec7413c3', # 123456
                                         'aeb80ba9fda44a8fad989017ee673149', # 123456
                                         '2d48f44fab344aefb37ccf610c037d93']} # 123456
# Half Cheetah with normal SAC
halfcheetah_sac_dict = {"Hidden_Dim 1x32" : ['4ca1884d14cd405ab43022e278bee411', # 4
                                             '8f62350abc7249a2803ad2844b83a3d7', # 5
                                             'c85ec113c4de4ec1a2a75da6ae042a3f',# 6
                                             '4a2d8c0b5321484aadb07eb1bf93b241'], # 7
                        "Hidden Dim 1x128" : ['53ff66634650478aa4f69838e2059810', # 1
                                              'e279da0b42b84962baf314653a9b866e', # 2
                                              '8277dfc87bd24a519ac43a6e6f01a974'], # 0
                        "Hidden Dim 2x256" : ['0f98107d6fcb4607b9d0e6e48cb648f6', # 123456
                                              'f9bf149914d34730b35be2c0171d010b', # 123456
                                              '327149f8f2ec456b8d58e27dca9e48a7']} # 123456
# Walker with normal SAC
walker_sac_dict = {"Hidden_Dim 1x32" : ['7df24994ed8e49caa2387590f8078b88', # 4
                                        '22a8cd3c7cdb44cabf6c8412c094b25e', # 5
                                        '1c6c50915d4746a09d223a8438341744', # 6
                                        '99fdeb425bbb4731a7226562561da257'], # 7
                   "Hidden Dim 1x128" : ['334a2c1ac59b4e57800b9dfe4e300061', # 123456
                                         '9f6e9c51995243ceadc37c3994988c31', # 2
                                         'c9c2b1c094db4651add96076decfeb4d'], # 1
                   "Hidden Dim 2x256" : ['d4ca797839a5412297288f67a1f14b85', # 123456
                                         'a426ae2bd095474e9316cdad581086c0', # 123456
                                         'c4a12a88d7674da2a6c8cafa0310f24c']} # 123456
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

# AR SAC Dictionaries to plot
# Hopper with autoregressive SAC
hopper_arsac_dict = {"Hidden_Dim 1x32" : ['8c8cc4849f9d4176a11b674dfd3757c9', # 4
                                          'f7181a2436764164bc721072cf18c695', # 5
                                          'b0e9331f7fc749fbbd8951a193bf33c5', # 6
                                          'ef321250ba1941069672f404f772d09d'], # 7
                     "Hidden Dim 1x128" : ['4dffe2e6c3204260893c8eb0b83e02a7', # 2
                                           'c67aacf2a30f49a8a5b2baa155355855', # 0
                                           '1489d7f3242d49e59892addbdc4c994a'], # 3
                     "Hidden Dim 2x256" : ['0bfd50fbd267460fad93d5f84607d35d', # 123456
                                           '2079085ac7e34ab99d429f2be1174cdb', # 123456
                                           'a89026b4717046eea379fb7abe32c2f5']} # 123456
# Half Cheetah with autoregressive SAC
halfcheetah_arsac_dict = {"Hidden_Dim 1x32" : ['20ae69065e884fc89c53593e90e6be81', # 4
                                               '66d0a2bd11514217a5ee56f15c003cfa', # 5
                                               '4fdcf5dcdaea4849809eac530a6cf9c0', # 6
                                               '72fd4939de164ca8b1d22268faf94c2d'], # 7
                          "Hidden Dim 1x128" : ['7642f7fb6edb49b3bf20a4969462ed97', # 1
                                                '1e725b1b8cf24632a9362bb68d47b6c3', # 2
                                                'c041e3e3611442ccb7dff13394562b83'], # 0
                          "Hidden Dim 2x256" : ['5ab8c92e50614dc898d4e3a5f1f46604', # 123456
                                                '5ca1185424a3442ebf8b6ace07244c8c', # 123456
                                                '0c5e41aee72944f294760aab96c4b99c']} # 123456
# Walker with autoregressive SAC
walker_arsac_dict = {"Hidden_Dim 1x32" : ['b9b47aea36cb4ac98adcaece37327447', # 7
                                          '1f6ea22efb0b4a6993876168168b9298', # 5
                                          '4858e72c7b624860a557e2f56fc9d8f2', # 6
                                          'de62274aad9746fcbc80458977217a6b'], # 4
                     "Hidden Dim 1x128" : ['6ea474869c3a47668d082290c6e2b6b7', # 123456
                                           'f23f84416a0348a18a84f6c7090fa5c6', # 1
                                           '548c8c937a17418ba94c4f552cb59329'], # 2
                     "Hidden Dim 2x256" : ['3bcc58d1e2644e5ebc8c847087064e4b', # 123456
                                           '91b1865bcbeb43dbb2f7730363e00ab8', # 123456
                                           'e557cf1ffa0d41a9a910606853ae2957']} # 123456
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
        "Walker2d-v2 w/ ARSAC" : walker_arsac_dict
        #"Ant-v2 w/ SAC" : ant_sac_dict,
        #"Ant-v2 w/ ARSAC" : ant_arsac_dict,
    }

if __name__ == "__main__":

    # Specify the folder we want to save the visualizations to
    base_dir = "paper_plots/"
    for env in to_plot_dict.keys():
        env_exp_dict = to_plot_dict[env]
        print("Visualizing ", env)

        plot_rewards(env_exp_dict, base_dir)