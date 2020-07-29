from comet_ml.api import API
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

matplotlib.style.use('seaborn')
# rc('text', usetex=True)
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

def make_log_scale_plots(experiment_dict, env_name):
    fig, axs = plt.subplots(1)
    #fig.suptitle(env_name + " AR use over time")
    axs.set_xlabel("Eval episodes")
    axs.set_ylabel("Avg. Log(Scale)")

    # Eval episode steps
    x_axis = range(1, 500)

    print(env_name + " Log Scale Plots")

    for key in experiment_dict.keys():
        print(key)
        # This is the final array we will plot
        avg_log_scales = np.zeros((len(x_axis), len(experiment_dict[key])))

        # For each experiment with this hidden dimension
        for i, experiment_id in enumerate(experiment_dict[key]):

            experiment = comet_api.get_experiment(project_name=project_name,
                                                  workspace=workspace,
                                                  experiment=experiment_id)
            asset_list = experiment.get_asset_list()

            # For every eval episode
            for step in x_axis:
                # Specify the file name
                fileName = "episode_step_" + str(step * 10000) + "_adj_scale"

                # Get the asset id
                match = [x for x in asset_list if x['fileName'] == fileName]
                if len(match) == 0:
                    continue
                asset_id = [x for x in asset_list if x['fileName'] == fileName][0]['assetId']

                # Now we get the asset
                scales = experiment.get_asset(asset_id, return_type="json")
                #print(scales)
                log_scale = np.mean(np.log(np.array(scales)))
                avg_log_scales[step - 1, i] += log_scale

        #print(avg_log_scales)
        mean = np.mean(avg_log_scales, axis=1)
        std = np.std(avg_log_scales, axis=1)

        axs.plot(x_axis, mean)
        axs.fill_between(x_axis, mean - std, mean + std, alpha=0.25)
    axs.legend(axs.get_lines(), experiment_dict.keys(), prop={'size': 10}, title="Hidden Dimension")
    fig.savefig("log_scale_plots/" + env_name.replace(" ", "_") + "_log_scale_over_training.pdf")


# Here we give the experiments to make our log scale plots
halfcheetah_dict = {"1x32": ['20ae69065e884fc89c53593e90e6be81',  # 4
                             '66d0a2bd11514217a5ee56f15c003cfa',  # 5
                             '4fdcf5dcdaea4849809eac530a6cf9c0',  # 6
                             '72fd4939de164ca8b1d22268faf94c2d'],  # 7
                    "1x128": ['84540c0c8b624f27a896215b1f9694d5',  # 4
                              '2d6669ef73204cf8b60e12425eaca729',  # 5
                              'b6aa62d9e6a64a5d826f0769ffc61541',  # 6
                              '3a825f923ae74495b92b23cc76d34c58'],  # 7
                    "2x256": ['9bd541c0c50a4d00af22733dc191e442',  # 4
                              '3b018186adfd42ada5aec213370e6fb9',  # 5
                              '411ca132f39642088104b4de18e10a4b',  # 6
                              '3e43d128046d4a6e98f8d2f527f10bea']}  # 7
walker_dict = {"Hidden_Dim 1x32": ['b9b47aea36cb4ac98adcaece37327447',  # 7
                                   '1f6ea22efb0b4a6993876168168b9298',  # 5
                                   '4858e72c7b624860a557e2f56fc9d8f2',  # 6
                                   'de62274aad9746fcbc80458977217a6b'],  # 4
               "Hidden Dim 1x128": ['eb7860b4d5034d9f8b1c398e131d8d46',  # 4
                                    'f0d2c250a85f4094bba26ce557599e49',  # 5
                                    '21d0132f5ac14405a2662749cc7e2b94',  # 6
                                    'c924443c1bb44ca79afe8eb911d2c613'],  # 7
               "Hidden Dim 2x256": ['2a98ce2c54a24735b3c79c3e6484eaff',  # 4
                                    'b14b367e4a184c22ac47079258358afc',  # 5
                                    'c402c66ad0454f5ab45936dd7b5923db',  # 6
                                    'caea315ef9f7486ea29ece089e2038e0']}  # 7
hopper_dict = {"Hidden_Dim 1x32": ['8c8cc4849f9d4176a11b674dfd3757c9',  # 4
                                   'f7181a2436764164bc721072cf18c695',  # 5
                                   'b0e9331f7fc749fbbd8951a193bf33c5',  # 6
                                   'ef321250ba1941069672f404f772d09d'],  # 7
               "Hidden Dim 1x128": ['9439535a4d2f4905b4e17000b6630b5b',  # 4
                                    '0d3d7f8ec09e442e9edfff15816bae57',  # 5
                                    '917e5d9a577f4fda8311f544e89e9bcc',  # 6
                                    'd9ef9d42944b44528390c9da3c2f27e3'],  # 7
               "Hidden Dim 2x256": ['e6717c467f124e128a7be1c0b4714724',  # 4
                                    'ed5e6ef37a1541e0adb9096b03f6f3e6',  # 5
                                    'f8d3661c1bc84e728dfdf83ac5b8770f',  # 6
                                    'e72b115236594b92a7f47328909fd5e3']}  # 7
# 07-22-20 tests
hopper_dict2 = {"Lookback 10": ['676b480ab1e8451d8d73f0f8c6b22bd0'],
                "Lookback 30": ['db4c0cee9aaf4725ad5caf9811eaceca']}

walker_dict2 = {"Lookback 10": ['0b4b4bd11e244a619ff34f07d812bab2'],
                "Lookback 30": ['2e320453ae1348dba90a81f8fb87df5c']}

halfcheetah_dict2 = {"Lookback 10": ['dec6edaf76714b2286b6eb601672d7d4'],
                     "Lookback 30": ['b8c6cb71130f402f94f9bf0c1b1202bd']}

if __name__ == "__main__":

    # This makes the plot showing how log(scale) changes over all eval episodes
    # make_log_scale_plots(halfcheetah_dict, "Half Cheetah")
    # make_log_scale_plots(walker_dict, "Walker")
    # make_log_scale_plots(hopper_dict, "Hopper")
    #make_log_scale_plots(ant_dict, "Ant")
    make_log_scale_plots(hopper_dict2, "07-22-20 Hopper")
    make_log_scale_plots(walker_dict2, "07-22-20 Walker")
    make_log_scale_plots(halfcheetah_dict2, "07-22-20 Half Cheetah")









