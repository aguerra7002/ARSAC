from comet_ml.api import API
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
matplotlib.style.use('seaborn')
#rc('text', usetex=True)
rc('font', family='serif')
import seaborn as sns

api_key = 'tHDbEydFQGW7F1MWmIKlEvrly'
workspace = 'aguerra'
project_name = 'arsac'
comet_api = API(api_key=api_key)

def get_mse(net1, net2):
    mse = 0
    for key in net1.keys():
        if "phi" in key and key in net2.keys():
            mse += torch.norm(net2[key] - net1[key], 2) / net2[key].numel()
    return mse

def get_net(exp_id, name):
    experiment = comet_api.get_experiment(project_name=project_name,
                                          workspace=workspace,
                                          experiment=exp_id)

    asset_list = experiment.get_asset_list()

    actor_asset_id = [x['assetId'] for x in asset_list if name == x['fileName']]
    if len(actor_asset_id) == 0:
        return None
    actor = experiment.get_asset(actor_asset_id[0])
    act_path = 'tmploaded/actor.model'
    with open(act_path, 'wb+') as f:
        f.write(actor)
    return torch.load(act_path)

def plot_flow_network_weights_change(base_exp_id, transfer_exp_ids, title):

    base_net = get_net(base_exp_id, "actor.model")
    mses = [[] for i in range(len(transfer_exp_ids))]
    for i, transfer_exp_id in enumerate(transfer_exp_ids):
        eval = 5
        while eval <= 295:
            name = "actor_eval_" + str(eval) + ".model"
            tr_net = get_net(transfer_exp_id, name)
            if tr_net is None:
                print("Missing", eval)
                mses[i].append(0)
            else:
                print(eval)
                mses[i].append(get_mse(base_net, tr_net))
            eval += 5
    # mses = np.array(mses)
    # mean = np.mean(mses, axis=1)
    # stds = np.std(mses, axis=1)
    for exp in range(len(mses)):
        plt.plot(mses[exp])
    # plt.fill_between(mean - stds, mean + stds, alpha=0.25)
    plt.savefig("transfer_network_change/jpgs/" + title + ".jpg")
    plt.savefig("transfer_network_change/pdfs/" + title + ".pdf")
    plt.clf()
walker = ("Walker Run AutoEnt 2x256", "113982d9ab8b4faf849ef5a2efb712f5", ['db603b5dabae46eea227c2232d4ec3a5',
                                                                            '5ff7fd4a1cdb4e57a33e6349f5e288ef',
                                                                            '1ffb2517534f4387982cfa4aae96baa1',
                                                                            'ff7ef746ccff4d3da3be6aced21ca938',
                                                                            '9a46b4a350b54610afcea3b04cd9fd1f'])

hopper = ("Hopper Hop AutoEnt 2x256", "e8218e27d9c74d659bc62b9242ab2de2", ['35ba36ee111b41929baf3a2591afce49',
                                                                            '598bd0127114433abe950a6eb3b08ebb',
                                                                            '7cba40d750c44d96bc0687e07e57f271',
                                                                            'f5a6a9ca7f534d7ba95ff7d61b7bc666',
                                                                            '7bc4b5be54a64416b5bedaea4ae820c6'])

quadruped = ("Quadruped Run AutoEnt 2x256", "a8f7092ff8de48cc97b7e8dd2e18849a", ['9574452c9b474fe99eb7efd5170e0c79',
                                                                                    '8c8800797c474819a7cec4265f63afdc', #<- this experiment may have incomplete data
                                                                                    '43037f0e4b47428cb1cf2be773ed7766',
                                                                                    '54fc235e8fbf4302b0d9065f020ef95c',
                                                                                    '2f97bea10b1d455e9b7179d88171488c'])

quadruped2 = ("Quadruped Run", "4e96ec38641a4ed59e31050e69811888", ['3a5ef39381d54d31a8d689bb4171beda',
                                                                    '9b980baa1e9d4d89af3b407e566947cf',
                                                                    '8538afd5185840e3a93d55c408b56f13'])
to_plot = [
    #walker,
    #hopper,
    quadruped2
]

if __name__ == "__main__":
    for tupl in to_plot:
        plot_flow_network_weights_change(tupl[1], tupl[2], tupl[0])