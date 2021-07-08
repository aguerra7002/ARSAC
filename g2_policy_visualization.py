from comet_ml.api import API
from arsac_experiment_analysis import run_eval_episode, setup_directory
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # TODO: Fix this hack by installing nomkl
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
    # print(type(start_step), type(end_step))
    x_axis = np.array(range(0, end_step))
    action_dim = eval_eps_dict[next(iter(eval_eps_dict.keys()))][0].shape[0]
    # For each action dimension (since we want one plot per dimension)
    for dim in range(action_dim):
        # Initialize the plot
        fig, axs = plt.subplots(nrows=len(eval_eps_dict.keys()), ncols=2)
        fig.set_size_inches(12, 3 * len(eval_eps_dict.keys()))
        fig.suptitle(title + ", Dim: " + str(dim))
        # For each eval episode
        for i, eval_ep in enumerate(eval_eps_dict.keys()):
            # Get the stats from the episode.
            base_mean, ar_mean, mean, sigma_means, reward = eval_eps_dict[eval_ep]

            # Make the plots
            axs[i, 0].plot(x_axis, np.tanh(base_mean[dim, 0: end_step]), label="SA action")
            axs[i, 0].plot(x_axis, np.tanh(ar_mean[dim, 0: end_step]), label="AR action")
            axs[i, 0].plot(x_axis, np.tanh(mean[dim, 0:end_step]), '.', color='black', label="final action")

            axs[i, 1].plot(x_axis, sigma_means[dim, 0:end_step], label="gate")
            axs[i, 1].set_ylim(-0.05, 1.05) # Make between 0 and 1

            # Formatting
            if i == len(eval_eps_dict.keys()) - 1:
                axs[i, 0].set_xlabel("Step")
                axs[i, 0].legend()
                axs[i, 1].set_xlabel("Step")
                axs[i, 1].legend()
            else:
                axs[i, 0].set_xticklabels([])
                axs[i, 1].set_xticklabels([])
            axs[i, 0].set_title("Eval Episode " + str(eval_ep) + ", Reward: " + str(reward))
            axs[i, 0].set_ylabel("Pre-Tanh Action")
            axs[i, 1].set_ylabel("Gate")

        # Save the figure
        fig.savefig(loc + "policy_vis_dim_" + str(dim))
        plt.clf()


# G2 2x256 Auto-entropy tuning with restricted policy deviation
exp_dict = {
    "Walker_Walk_2x256_Policy_Visualization": "790ec3d690d241c887004ba0523f3a22",
    "Walker_Run_2x256_Policy_Visualization": "1e8ee2b7e110481f8bd954417b64fb49",
    "Cheetah_Run2x256_Policy_Visualization": "481d7340b126484c9f771a5e4f27c635"
}


if __name__ == "__main__":

    for title in exp_dict.keys():
        arsac_exp_id = exp_dict[title]

        eval_eps_list = [65]#[5, 20, 50]
        # Need to build up the eval_eps_dict
        eval_eps_dict = {}
        os.chdir(setup_directory("g2_visualizations"))

        for eval_ep in eval_eps_list:
            if eval_ep == 300:
                # AKA final model
                actor_filename = "actor.model"
            else:
                actor_filename = "actor_eval_" + str(eval_ep) + ".model"

            _, ar_means, _, base_means, sigma_mean, rewards, _ = run_eval_episode(arsac_exp_id, title, actor_filename=actor_filename, prior_only=True, num_steps=1000)
            # Compute the pre-tanh means using the gate
            means = ar_means * sigma_mean + base_means * (1 - sigma_mean)
            reward = rewards.sum()
            print("Reward:", reward)
            eval_eps_dict[eval_ep] = (base_means, ar_means, means, sigma_mean, reward)
        os.chdir("../../")

        #make_g2_policy_visualization(eval_eps_dict, title=title)












