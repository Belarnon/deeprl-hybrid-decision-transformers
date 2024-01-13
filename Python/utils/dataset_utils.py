from matplotlib import pyplot as plt
import matplotlib.animation as animation
from typing import List, Any
from tqdm import tqdm
import numpy as np
import json

__TRAJECTORY_KEY = 'trajectories'
__TRANSITION_KEY = 'transitions'
__OBSERVATION_KEY = 'observation'
__CUMULATIVE_REWARD_KEY = 'reward'

# DATASET LOADING AND HANDLING
# ----------------------------

def load_dataset(dataset_path: str) -> dict:
    """
    Load a dataset from a json file.

    Args:
        dataset_path (str): Path to the dataset.

    Returns:
        dict: The dataset.
    """
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    return dataset

def save_dataset(dataset: dict, dataset_path: str) -> None:
    """
    Save a dataset to a json file.

    Args:
        dataset (dict): The dataset.
        dataset_path (str): The path to the dataset.
    """
    with open(dataset_path, 'w') as f:
        json.dump(dataset, f)

def extract_cumulative_reward_series(dataset: dict) -> np.ndarray:
    """
    Extract the cumulative reward series from the dataset.

    Args:
        dataset (dict): The dataset.

    Returns:
        np.ndarray: The cumulative reward series.
    """
    cumulative_rewards: List[float] = []
    trajectories: List[dict] = dataset[__TRAJECTORY_KEY]
    for trajectory in trajectories:
        transitions: List[dict] = trajectory[__TRANSITION_KEY]
        for transition in transitions:
            cumulative_rewards.append(transition[__CUMULATIVE_REWARD_KEY])
    
    return np.array(cumulative_rewards)

def compute_reward_deltas(cumulative_rewards: np.ndarray) -> np.ndarray:
    """
    Compute the reward deltas from the cumulative reward series.

    This calculates at each timestep t the reward delta r_t = R_t - R_{t-1}.

    Args:
        cumulative_rewards (np.ndarray): The cumulative reward series.

    Returns:
        np.ndarray: The reward deltas.
    """
    reward_deltas = np.zeros_like(cumulative_rewards)
    reward_deltas[1:] = cumulative_rewards[1:] - cumulative_rewards[:-1]
    reward_deltas[0] = cumulative_rewards[0]

    return reward_deltas

def extract_board(observation_vector: np.ndarray, board_shape: tuple = (10, 10)) -> np.ndarray:
    """
    Extract the board from the observation vector.

    Args:
        observation_vector (np.ndarray): The observation vector.
        board_shape (tuple, optional): The shape of the board. Defaults to (10, 10).

    Returns:
        np.ndarray: The board as a 2D array.
    """
    board_vector_length = board_shape[0] * board_shape[1]
    board_vector = observation_vector[:board_vector_length]
    board = board_vector.reshape(board_shape)
    return board

def extract_selection_block(observation_vector: np.ndarray, selection_block_index: int, board_shape: tuple = (10, 10), selection_block_shape: tuple = (5, 5)) -> np.ndarray:
    """
    Extract a selection block from the observation vector.

    Args:
        observation_vector (np.ndarray): The observation vector.
        selection_block_index (int): The index of the selection block.
        board_shape (tuple, optional): The shape of the board. Defaults to (10, 10).
        selection_block_shape (tuple, optional): The shape of the selection block. Defaults to (5, 5).

    Returns:
        np.ndarray: The selection block as a 2D array.
    """
    board_vector_length = board_shape[0] * board_shape[1]
    selection_block_vector_length = selection_block_shape[0] * selection_block_shape[1]
    selection_block_vector = observation_vector[board_vector_length + selection_block_index*selection_block_vector_length:board_vector_length + (selection_block_index+1)*selection_block_vector_length]
    selection_block = selection_block_vector.reshape(selection_block_shape)
    return selection_block

def extract_block_and_selection_series(dataset: dict, board_shape: tuple = (10, 10), selection_blocks: int = 3, selection_block_shape: tuple = (5, 5)) -> tuple:
    """
    Extract the block and selection series from the dataset.

    Args:
        dataset (dict): The dataset.
        board_shape (tuple, optional): The shape of the board. Defaults to (10, 10).
        selection_blocks (int, optional): The number of selection blocks. Defaults to 3.
        selection_block_shape (tuple, optional): The shape of the selection block. Defaults to (5, 5).

    Returns:
        tuple: The block series and the selection series.
    """
    board_series: List[np.ndarray] = []
    selection_block_series: List[List[np.ndarray]] = [[] for _ in range(selection_blocks)]
    trajectories: List[dict] = dataset[__TRAJECTORY_KEY]
    for trajectory in trajectories:
        transitions: List[dict] = trajectory[__TRANSITION_KEY]
        for transition in transitions:
            observation_vector: np.ndarray = np.array(transition[__OBSERVATION_KEY])
            board: np.ndarray = extract_board(observation_vector, board_shape)
            board_series.append(board)
            for selection_block_index in range(selection_blocks):
                selection_block: np.ndarray = extract_selection_block(observation_vector, selection_block_index, board_shape, selection_block_shape)
                selection_block_series[selection_block_index].append(selection_block)

    return board_series, selection_block_series

# DATASET MERGING
# ---------------

def merge_datasets(datasets: List[dict], verbose: bool = True) -> dict:
    """
    Merge multiple datasets into one.

    Args:
        datasets (List[dict]): The datasets.
        verbose (bool, optional): Whether to print the progress. Defaults to True.

    Returns:
        dict: The merged dataset.
    """
    merged_dataset = {
        __TRAJECTORY_KEY: []
    }
    for dataset in tqdm(datasets, disable=not verbose):
        trajectories: List[dict] = dataset[__TRAJECTORY_KEY]
        merged_dataset[__TRAJECTORY_KEY].extend(trajectories)
    return merged_dataset

def merge_datasets_files(dataset_paths: List[str], merged_dataset_path: str) -> None:
    """
    Merge multiple datasets into one and save it to a file.

    Args:
        dataset_paths (List[str]): The paths to the datasets.
        merged_dataset_path (str): The path to the merged dataset.
    """

    datasets = [load_dataset(dataset_path) for dataset_path in dataset_paths]
    merged_dataset = merge_datasets(datasets)
    save_dataset(merged_dataset, merged_dataset_path)


# DATASET PLOTTING
# ----------------

def plot_dataset_reward_timeseries(dataset: dict, title: str = None) -> None:
    """
    Plot the cumulative reward timeseries of the dataset.

    Args:
        dataset (dict): The dataset.
        title (str, optional): The title of the plot. Defaults to None.
    """
    cumulative_rewards = extract_cumulative_reward_series(dataset)
    plt.plot(cumulative_rewards)
    plt.xlabel('Transition')
    plt.ylabel('Cumulative Reward')
    if title is not None:
        plt.title(title)
    plt.show()

def plot_dataset_reward_deltas_timeseries(dataset: dict, title: str = None, plot_zero_line: bool = True, plot_errors: bool = True) -> None:
    """
    Plot the reward deltas timeseries of the dataset.

    Args:
        dataset (dict): The dataset.
        title (str, optional): The title of the plot. Defaults to None.
    """
    cumulative_rewards = extract_cumulative_reward_series(dataset)
    reward_deltas = compute_reward_deltas(cumulative_rewards)
    plt.plot(reward_deltas)
    plt.xlabel('Transition')
    plt.ylabel('Reward Delta')
    if title is not None:
        plt.title(title)
    if plot_zero_line:
        plt.axhline(y=0, color='k', linestyle='--')
    if plot_errors:
        # Mark the transitions where the reward delta is negative
        negative_reward_deltas = np.nonzero(reward_deltas < 0)[0]
        plt.scatter(negative_reward_deltas, reward_deltas[negative_reward_deltas], c='r', marker='x')
    plt.show()

def plot_histogram(values: np.ndarray, title: str = None, xlabel: str = None, ylabel: str = None, bins: int = 10) -> None:
    """
    Plot a histogram of the given values.

    Args:
        values (np.ndarray): The values.
        title (str, optional): The title of the plot. Defaults to None.
        xlabel (str, optional): The label of the x-axis. Defaults to None.
        ylabel (str, optional): The label of the y-axis. Defaults to None.
        bins (int, optional): The number of bins. Defaults to 10.
    """
    plt.hist(values, bins=bins)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()

def plot_reward_delta_histogram(dataset: dict, title: str = None, bins: int = 10) -> None:
    """
    Plot a histogram of the reward deltas of the dataset.

    Args:
        dataset (dict): The dataset.
        title (str, optional): The title of the plot. Defaults to None.
        bins (int, optional): The number of bins. Defaults to 10.
    """
    cumulative_rewards = extract_cumulative_reward_series(dataset)
    reward_deltas = compute_reward_deltas(cumulative_rewards)
    plot_histogram(reward_deltas, title=title, xlabel='Reward Delta', ylabel='Frequency', bins=bins)

def visualize_board_evolution(
        dataset: dict, 
        title: str = None, 
        board_shape: tuple = (10, 10),
        selection_blocks: int = 3, 
        selection_block_shape: tuple = (5, 5),
        play: bool = True
    ) -> animation.ArtistAnimation:
    """
    Visualize the evolution of the board over the course of the dataset.

    Args:
        dataset (dict): The dataset.
        title (str, optional): The title of the plot. Defaults to None.
        board_shape (tuple, optional): The shape of the board. Defaults to (10, 10).
        selection_blocks (int, optional): The number of selection blocks. Defaults to 3.
        selection_block_shape (tuple, optional): The shape of the selection blocks. Defaults to (5, 5).
        play (bool, optional): Whether to play the animation. Defaults to True.

    Returns:
        animation.ArtistAnimation: The animation.
    """

    # Extract the block and selection series
    board_series, selection_block_series = extract_block_and_selection_series(dataset, board_shape, selection_blocks, selection_block_shape)
    
    # Create a mosaic subplot animation with the booard large on top and the selection blocks small on the bottom
    fig, ax = plt.subplot_mosaic([
        ["board"]*selection_blocks,
        ["board"]*selection_blocks,
        ["board"]*selection_blocks,
        [f"selection_block_{i}" for i in range(selection_blocks)]
    ], layout='constrained', figsize=(7, 7))

    # Set the titles of the subplots
    ax["board"].set_title("Board")
    ax["board"].axis('off')
    for i in range(selection_blocks):
        ax[f"selection_block_{i}"].set_title(f"Selection Block {i}")
        ax[f"selection_block_{i}"].axis('off')

    # Create the animation
    ims = []
    for i in range(len(board_series)):
        im = []
        im.append(ax["board"].imshow(board_series[i], animated=True))
        for j in range(selection_blocks):
            im.append(ax[f"selection_block_{j}"].imshow(selection_block_series[j][i], animated=True))
        ims.append(im)
    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=1000)

    # Play the animation
    if play:
        plt.show()

    return ani


if __name__ == '__main__':
    dataset = load_dataset('dataset/evaluation/evaluation_1.json')
    visualize_board_evolution(dataset, title='Board Evolution')

    if False: 
        dataset_paths = [
            'dataset/demonstration_0.json',
            'dataset/demonstration_1.json',
            'dataset/game_session.json'
        ]

        merged_dataset_path = 'dataset/expert_trajectories.json'
        merge_datasets_files(dataset_paths, merged_dataset_path)
