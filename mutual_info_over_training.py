import torch
import numpy as np
import os
from torch.distributions import Normal
from arsac_experiment_analysis import run_eval_episode
from arsac_experiment_analysis import setup_directory
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
matplotlib.style.use('seaborn')
#rc('text', usetex=True)
rc('font', family='serif')
import seaborn as sns
sns.set_palette('Paired')

walker_run_base_dict4 = {
    "SAC": ['707acf6bfd714c6582e737db0191743f'],
    "ARSAC": ['cd2b9f003e404a8daff56dea22e0bcb3']
}

walker_run_base_dict5 = {
    "SAC" : ["3193d027636e4593a8ba71a84ee21638"],
    "ARSAC" : ["74e14df64e214d8ab68fcdb91bd44cb8"]
}

cheetah_run_base_dict5 = {
    "SAC": ["63585d83dafb49b99316d6814b4bbe03"],
    "ARSAC": ["2f23146b1baa45c8806a081cbfb9023e"]
}

to_plot_dict_2x256 = {
    "Walker Run AutoEnt 2x256HS": walker_run_base_dict4
}

to_plot_dict_1x32 = {
    #"Walker Run AutoEnt 1x32HS": walker_run_base_dict5,
    "Cheetah Run AutoEnt 1x32HS": cheetah_run_base_dict5
}

if __name__ == '__main__':
    sac_mi = []
    arsac_mi = []
    base_mi = []
    plot_dict = to_plot_dict_1x32
    x_axis = range(10, 251, 5)
    for key in plot_dict:
        os.chdir(setup_directory(key))
        base_dict = plot_dict[key]
        for ep in x_axis:
            print(key, ep)
            actor_filename = "actor_eval_" + str(ep) + ".model"
            sac_exp_id = base_dict["SAC"][0]
            arsac_exp_id = base_dict["ARSAC"][0]
            # Keep the means so that we can compute the base distribution log probs
            _, means, stds, _, _, rewards, log_probs = \
                    run_eval_episode(arsac_exp_id, key, plot_agent=False, eval=True, actor_filename=actor_filename)
            base_dist = Normal(torch.Tensor(means), torch.Tensor(stds))
            base_action = base_dist.rsample()
            base_log_probs = base_dist.log_prob(base_action).sum(0)

            _, _, _, _, _, sac_rewards, sac_log_probs = \
                    run_eval_episode(sac_exp_id, key, plot_agent=False, eval=True, actor_filename=actor_filename)
            # Compute the log probability for the base distribution of ARSAC. We do this by resampling,
            # which should be fine because this is averaged over 1000 steps.
            arsac_mi.append(np.mean(log_probs) - np.log(np.mean(np.exp(log_probs))))
            base_mi.append(torch.mean(-base_log_probs) - torch.log(torch.mean(torch.exp(-base_log_probs))))
            sac_mi.append(np.mean(sac_log_probs) - np.log(np.mean(np.exp(sac_log_probs))))
        # Now we should have all the mutual information stats for training
        print(base_mi, arsac_mi, sac_mi)
        plt.plot(x_axis, arsac_mi, label="ARSAC")
        plt.plot(x_axis, base_mi, label="ARSAC Base")
        plt.plot(x_axis, sac_mi, label="SAC")
        plt.legend()
        plt.xlabel("Eval Episode")
        plt.ylabel("Mutual Information")
        plt.title(key)
        plt.suptitle("Mutual Information Over Training")
        plt.savefig("./" + key.replace(" ", "_") + "_mutual_info.pdf")
        plt.savefig("./" + key.replace(" ", "_") + "_mutual_info.jpg")
        plt.clf()
        # Reset to original directory (go up two levels, ALWAYS)
        os.chdir("../../")
