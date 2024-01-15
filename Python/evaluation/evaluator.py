import torch
import wandb

import os
import argparse
import pyfiglet
from safetensors.torch import load_model
from random import randint

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from utils.training_utils import find_best_device
from utils.dataset_utils import plot_histogram
from evaluation.evaluate_episodes import evaluate_episode_rtg
from networks.decision_transformer import DecisionTransformer
import transformers

def banner():
    print(pyfiglet.figlet_format("HDT-Evaluator", font="slant"))

def parse_args():
    """
    Parses the arguments for the pretraining gym.
    """

    def boolean_type(v):
        """
        Helper function to parse boolean arguments.
        """

        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser(description="Trains the Hybrid Decision Transformer HDT")

    # MODEL
    def tuple_type(s):
        return tuple(map(int, s.split(',')))
    
    # MODEL
    parser.add_argument("-lmd", "--load_model_dir", type=str,
                        help="The filepath to the model that should be loaded.", required=True)
    parser.add_argument("-sdim", "--state_dim", type=int,
                        help="The dimension of the state vector in the given dataset.", default=175)
    parser.add_argument("-adim", "--act_dim", type=int,
                        help="The dimension of the action vector in the given dataset.", default=3)
    parser.add_argument("-maxlen", "--max_length", type=int,
                        help="The maximum number of past observations the transformer considers for the prediction.", required=True)
    parser.add_argument("-maxeplen", "--max_ep_len", type=int,
                        help="The (inclusive) maximum length of steps the evaluator takes.", default=4096)
    parser.add_argument("-aenc", "--act_enc", type=bool,
                        help="Flag to indicate whether the actions should be one-hot encoded.", default=True)
    parser.add_argument("-hd", "--hidden_dim", type=int,
                        help="The dimension of the embedding for the transformer.", default=128)
    parser.add_argument("-tanh", "--act_tanh", type=bool,
                        help="Set tanh layer at the end of action predictor.", default=False),
    parser.add_argument("-aspc", "--act_space", type=tuple_type,
                        help="The action space for the embedding if one-hot encoding is used.", default="3,10,10")
    parser.add_argument("-hugtrans", "--hugging_transformer", type=bool,
                        help="Flag to indicate whether the huggingface transformer should be used.", default=False)
    parser.add_argument("-stppcd", "--state_preprocessed", type=boolean_type,
                        help="Flag to indicate whether the state should be preprocessed.", default=True)
    parser.add_argument("-gridsz", "--grid_size", type=int,
                        help="The size of the grid in the state vector.", default=10)
    parser.add_argument("-blocksz", "--block_size", type=int,
                        help="The size of the blocks in the state vector.", default=5)

    # EVALUATION
    parser.add_argument("-trg_rt", "--target_return", type=float,
                        help="Return that should be achieved.", default=40.)
    parser.add_argument("-n_e", "--nr_episodes", type=int,
                        help="The number of evaluation episodes.", default=10)
    parser.add_argument("-gpu", "--use_gpu", type=bool,
                    help="Enable training on gpu device.", default=True)

    return parser.parse_args()

def evaluate():
    # cool banner and arguments    banner()
    args = parse_args()

    # setup
    device = find_best_device(args.use_gpu)

    # create and load the model
    # get action dim and action space
    action_space = args.act_space if args.act_enc else None
    action_dim = sum(action_space) if args.act_enc else args.act_dim
    if args.hugging_transformer:
        config = transformers.DecisionTransformerConfig(
            state_dim=args.state_dim,
            act_dim=action_dim,
            hidden_dim=args.hidden_dim,
            max_ep_len=args.max_ep_len,
            action_tanh=args.act_tanh
        )
        model = transformers.DecisionTransformerModel(config)
    else:
        model = DecisionTransformer(
            state_dim=args.state_dim,
            action_dim=action_dim,
            hidden_dim=args.hidden_dim,
            max_length=args.max_length,
            max_episode_length=args.max_ep_len,
            action_tanh=args.act_tanh,
            fancy_look_embedding=args.state_preprocessed,
            grid_size=args.grid_size,
            block_size=args.block_size
        )
    
    load_model(model, args.load_model_dir, strict=False)
    model.to(device)
    model.eval()

    # load the environment
    print("Waiting for Unity environment...")
    env = UnityEnvironment(seed=randint(0, 2**16))
    env = UnityToGymWrapper(env, allow_multiple_obs=True)
    print("Unity environment started successfully! Starting training...")

    # start evaluating loop
    with torch.no_grad():
        episodes_returns, episodes_lengths = evaluate_episode_rtg(
            env,
            args.state_dim,
            action_dim,
            model,
            device,
            target_return=40.,
            nr_episodes=args.nr_episodes,
            action_space=action_space,
            use_huggingface=args.hugging_transformer,
        )

    print(f"Average return and std: {episodes_returns.mean()},{episodes_returns.std()},\naverage length and std: {episodes_lengths.mean()},{episodes_lengths.std()}")

    plot_histogram(episodes_returns.cpu().numpy(), "Episode returns", xlabel="return", ylabel="frequency")
    plot_histogram(episodes_lengths.cpu().numpy(), "Episode lengths", xlabel="length", ylabel="frequency")

if __name__ == "__main__":
    evaluate()