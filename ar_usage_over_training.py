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

walker_walk_g3 = {
    "ARSAC": ["e3446c4a9f9c4ee5a1ad1713f462fb7e"]
}

to_plot_dict_2x256 = {
    "Walker Run AutoEnt 2x256HS": walker_run_base_dict4
}

to_plot_dict_1x32 = {
    "Walker Run AutoEnt 1x32HS": walker_walk_g3
    #"Cheetah Run AutoEnt 1x32HS": cheetah_run_base_dict5
}

if __name__ == '__main__':
    arsac_var = []
    plot_dict = to_plot_dict_1x32
    x_axis = range(10, 301, 20)
    for key in plot_dict:
        os.chdir(setup_directory(key))
        base_dict = plot_dict[key]
        for ep in x_axis:
            print(key, ep)
            actor_filename = "actor_eval_" + str(ep) + ".model"
            arsac_exp_id = base_dict["ARSAC"][0]
            # Keep the means so that we can compute the base distribution log probs
            _, _, shifts, _, _, _, _ = \
                    run_eval_episode(arsac_exp_id, key, plot_agent=False, eval=True, actor_filename=actor_filename)
            # This will hopefully give an indication for how much the ar_policy acts nontrivially on the policy over training.
            arsac_var.append(np.var(shifts))

        plt.plot(x_axis, arsac_var, label="ARSAC")
        plt.legend()
        plt.xlabel("Eval Episode")
        plt.ylabel("Variance of shift component")
        plt.title(key)
        plt.suptitle("ARSAC_Usage Over Training")
        plt.savefig("./" + key.replace(" ", "_") + "_ar_usage.pdf")
        plt.savefig("./" + key.replace(" ", "_") + "_ar_usage.jpg")
        plt.clf()
        # Reset to original directory (go up two levels, ALWAYS)
        os.chdir("../../")
