from comet_ml.api import API
from arsac_experiment_analysis import run_eval_episode, setup_directory
import numpy as np
import os
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
project_name = 'arsac_test'
comet_api = API(api_key=api_key)

# savgol filter
SMOOTH = True
WINDOW = 71
POLY_DEG = 3


"""
    Makes a plot showing how the g2 policy changes over training. Will create one plot for each action dimension.
    Each plot will contain N subplots showing the policy along the dimension at each of the N given eval episodes.
"""
def make_g2_policy_visualization(eval_eps_dict, start_step=0, end_step=180, title="G2 Policy Visualization"):
    # Create the appropriate directory
    loc = "g2_policy_visualizations/" + title + "/"
    if not os.path.exists(loc):
        os.makedirs(loc)

    # Range of steps to plot in a given episode
    print(type(start_step), type(end_step))
    x_axis = np.array(range(0, end_step))
    action_dim = eval_eps_dict[next(iter(eval_eps_dict.keys()))][0].shape[0]
    # For each action dimension (since we want one plot per dimension)
    for dim in range(action_dim):
        # Initialize the plot
        fig, axs = plt.subplots(len(eval_eps_dict.keys()))
        fig.set_size_inches(6, 3 * len(eval_eps_dict.keys()))
        fig.suptitle(title + ", Dim: " + str(dim))
        # For each eval episode
        for i, eval_ep in enumerate(eval_eps_dict.keys()):
            # Get the stats from the episode.
            base_mean, ar_mean, mean, reward = eval_eps_dict[eval_ep]

            # Make the plots
            axs[i].plot(x_axis, base_mean[dim, 0: end_step], label="SA action")
            axs[i].plot(x_axis, ar_mean[dim, 0: end_step], label="AR action")
            axs[i].plot(x_axis, mean[dim, 0:end_step], '.', color='black', label="final action")

            # Formatting
            if i == len(eval_eps_dict.keys()) - 1:
                axs[i].set_xlabel("Step")
                axs[i].legend()
            else:
                axs[i].set_xticklabels([])
            axs[i].set_title("Eval Episode " + str(eval_ep) + ", Reward: " + str(reward))
            axs[i].set_ylabel("Pre-Tanh Action")

        # Save the figure
        fig.savefig(loc + "policy_vis_dim_" + str(dim))
        plt.clf()




if __name__ == "__main__":

    # arsac_exp_id = "45ec5b0230214e93a5c7175889bb9b3b"
    # title = "Walker_Run_Test_Visualization"

    arsac_exp_id = "0c331921c58e462f83d193694ad98d76"
    title = "Cheetah_Run_Test_Visualization"

    eval_eps_list = [5, 25, 150, 290]
    # Need to build up the eval_eps_dict
    eval_eps_dict = {}
    os.chdir(setup_directory("g2_visualizations"))
    for eval_ep in eval_eps_list:
        if eval_ep == 300:
            # AKA final model
            actor_filename = "actor.model"
        else:
            actor_filename = "actor_eval_" + str(eval_ep) + ".model"

        _, ar_means, _, base_means, sigma_mean, rewards, _ = run_eval_episode(arsac_exp_id, title, actor_filename=actor_filename, prior_only=False, num_steps=1000)
        # Compute the pre-tanh means using the gate
        means = ar_means * sigma_mean + base_means * (1 - sigma_mean)
        reward = rewards.sum()
        eval_eps_dict[eval_ep] = (base_means, ar_means, means, reward)
    os.chdir("../../")

    make_g2_policy_visualization(eval_eps_dict, title=title)












