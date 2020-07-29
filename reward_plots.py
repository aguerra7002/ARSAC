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

# Normal SAC Dictionaries to plot
# Hopper with normal SAC
hopper_sac_dict = {"SAC (1x32)" : ['f675f9f430404111a9530dfb4d276273', # 4
                                        'ca092a2dd00e48008d43e068aec3ad72', # 5
                                        '0223b49a689b4ab190134ca519eaceb6', # 6
                                        '8efbdefa14da4cc387c7dc9649c57943'], # 7
                   "SAC (1x128)" : ['bbf32b48623344d0bfacdf4e7d4372a4', # 4
                                         '686cc76e8f1b487a990b2c46f398805c', # 5
                                         '215831a65ef3472c9d70bd64af8cbf97', # 6
                                         '37e3ea9d5ec44469b65a029d775c27b6'], #7
                   "SAC (2x256)" : ['df4b5dcf96574f77bbc550fd059d6d4a', # 4
                                         'e287a511ec574820a8d65445f53860d4', # 5
                                         'f3f489616594451a90d1aac4ee6fc341', # 6
                                         '8a04435f79044ebd991bb224b602a3a3'], # 7
                   "ARSAC (1x32)" : ['8c8cc4849f9d4176a11b674dfd3757c9', # 4
                                          'f7181a2436764164bc721072cf18c695', # 5
                                          'b0e9331f7fc749fbbd8951a193bf33c5', # 6
                                          'ef321250ba1941069672f404f772d09d'], # 7
                   "ARSAC (1x128)" : ['9439535a4d2f4905b4e17000b6630b5b', # 4
                                           '0d3d7f8ec09e442e9edfff15816bae57', # 5
                                           '917e5d9a577f4fda8311f544e89e9bcc', # 6
                                           'd9ef9d42944b44528390c9da3c2f27e3']} # 7
                   # "ARSAC (2x256)" : ['e6717c467f124e128a7be1c0b4714724', # 4
                   #                         'ed5e6ef37a1541e0adb9096b03f6f3e6', # 5
                   #                         'f8d3661c1bc84e728dfdf83ac5b8770f', # 6
                   #                         'e72b115236594b92a7f47328909fd5e3']} # 7
# Half Cheetah with normal SAC
halfcheetah_sac_dict = {"SAC (1x32)" : ['4ca1884d14cd405ab43022e278bee411', # 4
                                             '8f62350abc7249a2803ad2844b83a3d7', # 5
                                             'c85ec113c4de4ec1a2a75da6ae042a3f',# 6
                                             '4a2d8c0b5321484aadb07eb1bf93b241'], # 7
                        "SAC (1x128)" : ['91ccba1dc34a44959e8f65cfbdc54cbc', # 4
                                              'ab1d8b77c2cb438eab56c4c0fad9f041', # 5
                                              '3e8a4d8153414dbcb6d2bac12b5215e8', # 6
                                              '59123c9d68b544d8be362ea57e3dd5e4'], # 7
                        "SAC (2x256)" : ['230350690f1e4cb5a3aa7738d44b9979', # 4
                                              '84aa71e59525445d82e72ef833a90526', # 5
                                              '2b7f087484064ca4ad2ace50528afc28', # 6
                                              '3eade1f457f74cbd8b9107259264cd99'], # 7
                        "ARSAC (1x32)" : ['20ae69065e884fc89c53593e90e6be81', # 4
                                               '66d0a2bd11514217a5ee56f15c003cfa', # 5
                                               '4fdcf5dcdaea4849809eac530a6cf9c0', # 6
                                               '72fd4939de164ca8b1d22268faf94c2d'], # 7
                        "ARSAC (1x128)" : ['84540c0c8b624f27a896215b1f9694d5', # 4
                                                '2d6669ef73204cf8b60e12425eaca729', # 5
                                                'b6aa62d9e6a64a5d826f0769ffc61541', # 6
                                                '3a825f923ae74495b92b23cc76d34c58']} # 7
                        # "ARSAC (2x256)" : ['9bd541c0c50a4d00af22733dc191e442', # 4
                        #                         '3b018186adfd42ada5aec213370e6fb9', # 5
                        #                         '411ca132f39642088104b4de18e10a4b', # 6
                        #                         '3e43d128046d4a6e98f8d2f527f10bea']} # 7
# Walker with normal SAC
walker_sac_dict = {"SAC (1x32)" : ['7df24994ed8e49caa2387590f8078b88', # 4
                                        '22a8cd3c7cdb44cabf6c8412c094b25e', # 5
                                        '1c6c50915d4746a09d223a8438341744', # 6
                                        '99fdeb425bbb4731a7226562561da257'], # 7
                   "SAC (1x128)" : ['b0798d9da9cc46c19dc9829249069713', # 4
                                         '171843d852fb406caab00ec2af58b65d', # 5
                                         '0d6ac53a5d724e0cb3091b42bd0c6fb8', # 6
                                         'd30d1373d02c4b8c91fedc8c41e0af08'], # 7
                   "SAC (2x256)" : ['e9bd85dd73d54c0a975ca3266eb9874e', # 4
                                         '5e11ed420304458ca972b03183bca432', # 5
                                         '8c69453fc61c4fbabd7a1fe1bc538427', # 6
                                         '6556fad8190d4e099ce752858ea1ecd2'], # 7
                   "ARSAC (1x32)" : ['b9b47aea36cb4ac98adcaece37327447', # 7
                                          '1f6ea22efb0b4a6993876168168b9298', # 5
                                          '4858e72c7b624860a557e2f56fc9d8f2', # 6
                                          'de62274aad9746fcbc80458977217a6b'], # 4
                   "ARSAC (1x128)" : ['eb7860b4d5034d9f8b1c398e131d8d46', # 4
                                           'f0d2c250a85f4094bba26ce557599e49', # 5
                                           '21d0132f5ac14405a2662749cc7e2b94', # 6
                                           'c924443c1bb44ca79afe8eb911d2c613']} # 7
                     # "ARSAC (2x256)" : ['2a98ce2c54a24735b3c79c3e6484eaff', # 4
                     #                       'b14b367e4a184c22ac47079258358afc', # 5
                     #                       'c402c66ad0454f5ab45936dd7b5923db', # 6
                     #                       'caea315ef9f7486ea29ece089e2038e0']} # 7
# Ant with normal SAC
ant_sac_dict = {"SAC (1x32)" : ['826e53dbaf2e440daeab39e2960c8107',
                                     'a9f411d5777f40ccbe2fcf8dda73dee4',
                                     '013bd766d019463e86e7969054f437c8',
                                '3af76a80a3bf405b946f52035058be87'],
                "SAC (1x128)" : ['7cb642106ffb4d7b83d9a693a90e6316',
                                      '059961b74bf346eb9c04726eb0ab5f9e',
                                      '58d1fa05ce6349a8aeab669fe58b3f29',
                                 'fd4296d0f8074164bdf0ce10bea521b9'],
                "SAC (2x256)" : ['607e993a4259458b9846593c838d84e1',
                                      '6b5404bc3299499d9ba331cf2bad32b7',
                                      'b498a149d8754294a5f510b151d40e85',
                                 '41cb97ae2e6d4eb59cd936cb801556e9'],
                "ARSAC (1x32)" : ['09dfd4bb189646a3bc4294851d274510',
                                       '9641b2394bb041468843de435e868635',
                                       '1b6015332168429c938515e22adebd14',
                                  '8c24d64f57ac436a86231e31006dc926'],
                "ARSAC (1x128)" : ['c47e13563a9c47febd90e6b561fd7185',
                                        '9c3459b21fb0478e93be72553c527957',
                                        'db1ab8cd82754f7e85fa9afd59754c26',
                                   'd3330aac74af441bbe3d33506040475e']}
                # "ARSAC (2x256)" : ['',
                #                         '',
                #                         '']}

temp_dict = {"ARSAC (1 x 32)" : ['ce91f8efa4674be0b09ea2e844c3771b']}

# 07-22-20 tests
hopper_dict2 = {"Lookback 10": ['676b480ab1e8451d8d73f0f8c6b22bd0'],
                "Lookback 30": ['db4c0cee9aaf4725ad5caf9811eaceca']}

walker_dict2 = {"Lookback 10": ['0b4b4bd11e244a619ff34f07d812bab2'],
                "Lookback 30": ['2e320453ae1348dba90a81f8fb87df5c']}

halfcheetah_dict2 = {"Lookback 10": ['dec6edaf76714b2286b6eb601672d7d4'],
                     "Lookback 30": ['b8c6cb71130f402f94f9bf0c1b1202bd']}

to_plot_dict = {
        # "Hopper-v2" : hopper_sac_dict,
        # "HalfCheetah" : halfcheetah_sac_dict,
        # "Walker2d" : walker_sac_dict,
        # "Ant" : ant_sac_dict
        #"ce91f8efa4674be0b09ea2e844c3771b" : temp_dict
        "07-22-20 Hopper" : hopper_dict2,
        "07-22-20 Walker" : walker_dict2,
        "07-22-20 Half-Cheetah" : halfcheetah_dict2
    }


if __name__ == "__main__":

    # Specify the folder we want to save the visualizations to
    base_dir = "reward_plots/"
    for env in to_plot_dict.keys():
        env_exp_dict = to_plot_dict[env]
        print("Visualizing ", env)

        plot_rewards(env_exp_dict, base_dir)