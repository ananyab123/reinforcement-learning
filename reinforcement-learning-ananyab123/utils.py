from IPython.display import display, clear_output
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.cm as cm
from typing import List, Dict, Tuple, DefaultDict, Optional, Any
from collections import defaultdict
import gymnasium as gym

BlackjackState = Tuple[int, int, int] # player hand total, dealer's card value, either 0 or 1 depending on the presence of a usable ace
BlackjackAction = int # either 0 to stand or 1 to hit
BlackjackPolicy = Dict[BlackjackState, BlackjackAction] # mapping each state to an action

ValueFunction = Dict[BlackjackState, float] # for Part 3, a table that maps each state to a value

QTable = Dict[Tuple[BlackjackState, BlackjackAction], float] # for Part 4, a table that maps each state and action pair to a value


def render_gym_blackjack_env(env: gym.Env, state: BlackjackState) -> None:
    """
    Render and display the current state of a Gymnasium Blackjack environment.
    
    Args:
        env: Gymnasium environment with render capability
        state: Current blackjack state (player_total, dealer_card, usable_ace)
    """
    rgb_array = env.render()
    image = PIL.Image.fromarray(rgb_array)
    image = image.resize((image.width // 2, image.height // 2))
    image.save('blackjack.png')
    display(image)
    print(f"Player Total: {state[0]}, Dealer's Card: {state[1]}, Usable Ace: {state[2]}")

def Q_table_to_policy(Q: QTable) -> BlackjackPolicy:
    """
    Convert a Q-table to a deterministic policy by selecting greedy actions.
    
    For each state, this function selects the action with the highest Q-value.
    
    Args:
        Q: Q-table mapping (state, action) pairs to Q-values
        
    Returns:
        Policy dictionary mapping states to optimal actions
    """
    Q_numpy = np.zeros((18, 10, 2, 2))
    states = [(player, dealer, ace) for player in range(4, 22) for dealer in range(1, 11) for ace in range(2)]

    for state_action_pair in Q:
        state, action = state_action_pair
        player_total, dealer_card, usable_ace = state
        Q_numpy[player_total - 4, dealer_card - 1, usable_ace, action] = Q[state_action_pair]

    Q_policy = np.argmax(Q_numpy, axis=-1)
    pi = {}
    for state in states:
        player_total, dealer_card, usable_ace = state
        pi[state] = Q_policy[player_total - 4, dealer_card - 1, usable_ace]
    return pi

def plot_value_function(V: ValueFunction) -> None:
    """
    Plot two side-by-side heatmaps of the value function for states with and without a usable ace.
    
    This visualization helps understand the optimal strategy by showing the expected
    long-term reward for each state. Higher values (brighter colors) indicate
    more favorable positions for the player.

    Args:
        V: Value function dictionary mapping states to expected values
    """
    V_numpy = np.zeros((18, 10, 2))
    for state in V:
        player_total, dealer_card, ace = state
        V_numpy[player_total - 4, dealer_card - 1, ace] = V[state]
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    for idx, usable_ace in enumerate([0, 1]):
        ax = axs[idx]
        im = ax.imshow(V_numpy[:, :, usable_ace], aspect='auto', origin='lower', cmap='viridis')
        
        ax.set_xlabel("Dealer's Card")
        ax.set_ylabel("Player's Sum")
        ax.set_xticks(ticks=np.arange(10))
        ax.set_xticklabels([str(i) for i in range(1, 11)])
        ax.set_yticks(ticks=np.arange(18))
        ax.set_yticklabels([str(i) for i in range(4, 22)])
        
        word = "with" if usable_ace else "without"
        ax.set_title(f'Value Function for States {word} a Usable Ace')

    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), label='State Value', fraction=0.02, pad=0.04)

    plt.show()

def plot_policy(policy: BlackjackPolicy) -> None:
    """
    Plot two side-by-side heatmaps of the Blackjack policy for states with and without a usable ace.
    
    Color coding:
    - Red: Stay/Stand (action 0)
    - Green: Hit (action 1)
    
    This visualization shows the optimal action to take in each state,
    providing insights into when it's better to be conservative vs. aggressive.

    Args:
        policy: Policy dictionary mapping states to actions (0=stay, 1=hit)
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    for idx, usable_ace in enumerate([0, 1]):
        policy_matrix = np.zeros((18, 10))

        for state in policy:
            player_total, dealer_card, ace = state
            if ace == usable_ace:
                policy_matrix[player_total - 4, dealer_card - 1] = policy[state]

        ax = axs[idx]
        im = ax.imshow(policy_matrix, cmap=ListedColormap(['red', 'green']), origin='lower')
        ax.set_xlabel("Dealer's Card")
        ax.set_xticks(ticks=np.arange(10))
        ax.set_xticklabels([str(i) for i in range(1, 11)])
        ax.set_yticks(ticks=np.arange(18))
        ax.set_yticklabels([str(i) for i in range(4, 22)])
        ax.set_ylabel("Player's Total")

        word = "with" if usable_ace else "without"
        ax.set_title(f"Best Actions (Red: Stay, Green: Hit) {word} Usable Ace")

    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), ticks=[0, 1], label='Action (0: Stay, 1: Hit)', fraction=0.02, pad=0.04)

    plt.show()

def plot_observation_count(observation_counts: Dict[Tuple[BlackjackState, BlackjackAction], int]) -> None:
    """
    Plot two heat maps side-by-side of log-scaled observation counts for each state-action pair.
    
    This visualization shows how frequently different states were visited during
    Q-learning, helping to identify potential exploration issues. Darker areas
    indicate rarely visited states that may have unreliable Q-values.

    Args:
        observation_counts: Dictionary mapping (state, action) pairs to visit counts
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    
    for idx, usable_ace in enumerate([0, 1]):
        count_matrix = np.zeros((18, 10))

        for state_action, count in observation_counts.items():
            player_total, dealer_card, ace = state_action[0]
            if ace == usable_ace:
                count_matrix[player_total - 4, dealer_card - 1] = count

        log_count_matrix = np.log1p(count_matrix)  # log(1 + count) for better scaling

        ax = axs[idx]
        im = ax.imshow(log_count_matrix, cmap='viridis', aspect='auto', origin='lower')
        ax.set_xlabel("Dealer's Card")
        ax.set_ylabel("Player's Total")
        ax.set_xticks(ticks=np.arange(10))
        ax.set_xticklabels([str(i) for i in range(1, 11)])
        ax.set_yticks(ticks=np.arange(18))
        ax.set_yticklabels([str(i) for i in range(4, 22)])
        
        word = "with" if usable_ace else "without"
        ax.set_title(f"Log-scaled Observation Count {word} Usable Ace")

    cbar_ax = fig.add_axes([1, 0.17, 0.02, 0.7]) 
    fig.colorbar(im, cax=cbar_ax, label='Log-scaled Observation Count')
    
    plt.tight_layout()
    plt.show()
    
def plot_policy_with_observation(observation_counts: Dict[Tuple[BlackjackState, BlackjackAction], int], 
                               policy: BlackjackPolicy) -> None:
    """
    Plot policy heatmaps with confidence indicated by opacity based on observation frequency.
    
    This combines policy visualization with exploration statistics:
    - Color indicates action: red=stand, green=hit
    - Opacity indicates confidence: more opaque = more frequently visited = higher confidence
    
    States with low opacity may have unreliable policy recommendations due to
    insufficient exploration during Q-learning.

    Args:
        observation_counts: Dictionary mapping (state, action) pairs to visit counts
        policy: Policy dictionary mapping states to actions
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    
    norm = Normalize(vmin=0, vmax=1)
    
    for idx, usable_ace in enumerate([0, 1]):
        action_matrix = np.zeros((18, 10))  
        opacity_matrix = np.zeros((18, 10))  

        max_count = max(observation_counts.values(), default=1)
        
        for state_action, count in observation_counts.items():
            player_total, dealer_card, ace = state_action[0]
            if ace == usable_ace:
                action = policy[state_action[0]] 
                action_matrix[player_total - 4, dealer_card - 1] = action
                
                log_opacity = np.log1p(count) / np.log1p(max_count)  
                opacity_matrix[player_total - 4, dealer_card - 1] = max(log_opacity, 0.1)

        rgba_matrix = np.zeros((18, 10, 4))
        for i in range(18):
            for j in range(10):
                if action_matrix[i, j] == 0:  
                    rgba_matrix[i, j] = [0.6, 0, 0, opacity_matrix[i, j]]
                elif action_matrix[i, j] == 1:  
                    rgba_matrix[i, j] = [0, 0.5, 0, opacity_matrix[i, j]]

        ax = axs[idx]
        ax.imshow(rgba_matrix, origin='lower')
        ax.set_xlabel("Dealer's Card")
        ax.set_ylabel("Player's Total")
        ax.set_xticks(ticks=np.arange(10))
        ax.set_xticklabels([str(i) for i in range(1, 11)])  # Fixed: dealer cards are 1-10, not 2-12
        ax.set_yticks(ticks=np.arange(18))
        ax.set_yticklabels([str(i) for i in range(4, 22)])
        
        word = "with" if usable_ace else "without"
        ax.set_title(f"Policy Heatmap {word} Usable Ace")

    cax_red = fig.add_axes([0.37, 0.07, 0.25, 0.02]) 
    sm_red = cm.ScalarMappable(cmap='Reds', norm=norm)
    sm_red.set_array([])
    cbar_red = plt.colorbar(sm_red, cax=cax_red, orientation='horizontal')
    cbar_red.set_label('Stand - Log-scaled Confidence')

    cax_green = fig.add_axes([0.37, -0.03, 0.25, 0.02])  
    sm_green = cm.ScalarMappable(cmap='Greens', norm=norm)
    sm_green.set_array([])
    cbar_green = plt.colorbar(sm_green, cax=cax_green, orientation='horizontal')
    cbar_green.set_label('Hit - Log-scaled Confidence')

    plt.tight_layout(rect=[0, 0.1, 1, 1])  
    plt.show()

# def plot_policy_performance(mdp):
#     mdp = Blackjack(gym.make('Blackjack-v1', render_mode='rgb_array'))
#     policies = []
#     for num_episodes in [100, 1000, 10000, 100000, 1000000]:
#         Q, _ = q_learning(mdp = mdp, num_episodes = num_episodes, alpha=0.01, gamma=1.0)
#         pi = Q_table_to_policy(Q)
#         policies.append(pi)
        
#         win_rates, avg_rewards = [], []

#     # Run Monte Carlo simulations for each policy in the list
#     for policy in policies:
#         win_rate, avg_reward = monte_carlo_simulation(mdp, policy, 100000)
#         win_rates.append(win_rate)
#         avg_rewards.append(avg_reward)
    
#     iterations = [100, 1000, 10000, 100000, 1000000]

#     # Plot win rates
#     plt.figure(figsize=(12, 5))

#     plt.subplot(1, 2, 1)
#     plt.plot(iterations, win_rates, label="Q-Learning", color='blue', marker='o')
#     plt.xlabel("Iterations")
#     plt.ylabel("Win Rate")
#     plt.title("Win Rate Across Iterations")
#     plt.legend()

#     # Plot average rewards
#     plt.subplot(1, 2, 2)
#     plt.plot(iterations, avg_rewards, label="Q-Learning", color='blue', marker='o')
#     plt.xlabel("Iterations")
#     plt.ylabel("Average Reward")
#     plt.title("Average Reward Across Iterations")
#     plt.legend()

#     plt.tight_layout()
#     plt.show()