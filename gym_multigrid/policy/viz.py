import gymnasium as gym
from gym_multigrid.utils.misc import save_frames_as_gif
from gym_multigrid.core.agent import AgentT
import torch
import matplotlib
import matplotlib.pyplot as plt


def generate_policy_heatmap_with_arrows(
        policy: AgentT, 
        env: gym.Env, 
        filename: str, 
        device: str = "cpu"
    ) -> tuple[torch.Tensor, list[plt.patches.FancyArrowPatch]]:
    """
    Generates a heatmap representing the state-action values with arrows indicating the highest value action.
    Also saves an image of the rendered env.

    Args:
        policy (nn.Module): The policy network.
        env (gym.Env): The gym environment instance.
        filename (str): String to use for naming the saved plot file.
        device (str, optional): Device to use for computations ("cpu" or "cuda"). Defaults to "cpu".

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The 2D tensor representing the heatmap of state-action values.
            - list: A list of plt.patches.FancyArrowPatch objects for each state.
    """
    # get the initial state
    s, _ = env.reset()
    states = env.get_toroids()
    state = states[0]
    save_frames_as_gif([env.render()], filename=filename)

    # Get environment dimensions
    state_space_shape = (env.grid.width, env.grid.height)

    # Initialize heatmap with zeros
    heatmap = torch.zeros(state_space_shape, device=device)
    arrows = []

    # Iterate through all possible states in the grid
    for idx in range(state_space_shape[0] * state_space_shape[1]):
        x = idx // env.grid.width
        y = idx % env.grid.width
        state = env.toroid(idx)

        # Get state-action values for the current state
        state_action_values = policy.get_qvals(state)

        action_space_size = env.action_space.n
        state_action_values = state_action_values.reshape(1, action_space_size)

        # Find highest action value and corresponding index
        max_value, max_index = torch.max(state_action_values, dim=1)

        # Populate heatmap with highest action value
        heatmap[x, y] = max_value

        # Determine arrow direction based on highest action index
        arrow_tail = (y, x)  # Invert coordinates for matplotlib convention
        if max_index == 0:  # up
            arrow_head = (y, x + 0.3)
        elif max_index == 1:  # right
            arrow_head = (y + 0.3, x)
        elif max_index == 2:  # down
            arrow_head = (y, x - 0.3)
        else:  # left
            arrow_head = (y - 0.3, x)

        # Create arrow patch
        arrow = matplotlib.patches.FancyArrowPatch(
            arrow_tail, arrow_head, mutation_scale=15, color="red"
        )
        arrows.append(arrow)

    return heatmap, arrows


def plot_heatmap_with_arrows(
        heatmap: torch.Tensor, 
        arrows: list[plt.patches.FancyArrowPatch], 
        filename: str, 
        figsize: tuple[int, int] = (8, 8)
    ) -> None:
    """
    Plots the heatmap with arrows on top.

    Args:
        heatmap (torch.Tensor): The 2D tensor representing the heatmap.
        arrows (list): A list of matplotlib.patches.FancyArrowPatch objects.
        filename (str): String to use for naming the saved plot file.
        figsize (tuple, optional): Figure size for the plot. Defaults to (8, 8).
    """

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(heatmap.cpu().numpy())
    for arrow in arrows:
        ax.add_patch(arrow)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("State Value", rotation=-90, va="bottom")
    ax.set_xlabel("X-Coordinate")
    ax.set_ylabel("Y-Coordinate")
    plt.savefig(filename + "vmap-arrows.png")
    plt.show()


def plot_heatmap(
        heatmap: torch.Tensor, 
        filename: str, 
        figsize: tuple[int, int] = (8, 8)
    ) -> None:
    """
    Plots the heatmap.

    Args:
        heatmap (torch.Tensor): The 2D tensor representing the heatmap.
        filename (str): String to use for naming the saved plot file.
        figsize (tuple, optional): Figure size for the plot. Defaults to (8, 8).
    """

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(heatmap.cpu().numpy())
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("State Value", rotation=-90, va="bottom")
    ax.set_xlabel("X-Coordinate")
    ax.set_ylabel("Y-Coordinate")
    plt.savefig(filename + "vmap.png")
    plt.show()
