import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchrl.data.replay_buffers import ReplayBuffer, ListStorage
from tensordict import TensorDict

import os
from random import randint
import argparse
import pyfiglet
from safetensors.torch import save_model, load_model
from tqdm import tqdm

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

import transformers
from transformers.models.decision_transformer.modeling_decision_transformer import DecisionTransformerOutput

from dataset.trajectory_dataset import TrajectoryDataset
from evaluation.evaluate_episodes import evaluate_episode_rtg
from modules.loss.action_crossentropy import TenTenCEActionLoss
from modules.loss.action_loglikelihood import TenTenNLLActionLoss
from networks.online_decision_transformer import OnlineDecisionTransformer
from utils.training_utils import find_best_device, encode_actions, setup_wandb
from utils.replay_buffers.list_replay_buffer import ListReplayBuffer

def banner():
    print(pyfiglet.figlet_format("HDT-Online Trainer", font="slant"))

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
    # REPLAY BUFFER
    parser.add_argument("-ds", "--dataset", type=str,
                        help="The dataset to use as training data.", required=True)
    parser.add_argument("-dsval", "--dataset_validation", type=str,
                        help="The dataset to use as validation data.", default=None)
    parser.add_argument("-rbsz", "--replay_buffer_size", type=int,
                        help="The size of the replay buffer.", default=100)
    parser.add_argument("-b", "--batch_size", type=int,
                        help="The batch size to use for training.", default=16)
    # TRANSFORMER
    def tuple_type(s):
        return tuple(map(int, s.split(',')))
    parser.add_argument("-sdim", "--state_dim", type=int,
                        help="The dimension of the state vector in the given dataset.", default=175)
    parser.add_argument("-adim", "--act_dim", type=int,
                        help="The dimension of the action vector in the given dataset.", default=3)
    parser.add_argument("-maxeplen", "--max_ep_len", type=int,
                        help="The (inclusive) maximum length of ANY sequences to train on. This is used for the timestep embedding.", default=4096)
    parser.add_argument("-aenc", "--act_enc", type=bool,
                        help="Flag to indicate whether the actions should be one-hot encoded.", default=True)
    parser.add_argument("-aspc", "--act_space", type=tuple_type,
                        help="The action space for the embedding if one-hot encoding is used.", default="3,10,10")
    parser.add_argument("-hd", "--hidden_dim", type=int,
                        help="The dimension of the embedding for the transformer.", default=128)
    parser.add_argument("-tanh", "--act_tanh", type=bool,
                        help="Set tanh layer at the end of action predictor.", default=False),
    parser.add_argument("-stppcd", "--state_preprocessed", type=boolean_type,
                        help="Flag to indicate whether the state should be preprocessed.", default=True)
    parser.add_argument("-gridsz", "--grid_size", type=int,
                        help="The size of the grid in the state vector.", default=10)
    parser.add_argument("-blocksz", "--block_size", type=int,
                        help="The size of the blocks in the state vector.", default=5)
    parser.add_argument("-xform", "--use_xformers", type=boolean_type,
                        help="Flag to indicate whether the xFormers encoder should be used.", default=False)
    
    # TRAINING
    #   LOAD AND SAVE
    parser.add_argument("-lm", "--load_model", type=bool,
                        help="Load the model", default=False)
    parser.add_argument("-lmd", "--load_model_dir", type=str,
                        help="The filepath to the model that should be loaded.", required=True)
    
    #   TRAINING ARGUMENTS
    
    parser.add_argument("-lr", "--learning_rate", type=float,
                        help="The learning rate to use for training.", default=1e-4)
    parser.add_argument("-ss", "--step_size", type=int,
                        help="The stepsize used in the SGD algorithm. ", default=1)
    parser.add_argument("-dc", "--lr_decay", type=float,
                        help="Decay of the learning rate for a Learning Rate Scheduler.", default=1.0)
    parser.add_argument("-lfn", "--loss_fn", type=str,
                        help="Name of loss function to use.", default="NLL")
    parser.add_argument("-maxonit", "--max_online_iterations", type=int,
                        help="The maximum number of iterations for online training.", default=10) # low default value for testing
    parser.add_argument("-k", "--context_length", type=int,
                        help="The context length (aka max_length) for the transformer.", default=30)

    #   MISCELLANEOUS
    parser.add_argument("-w", "--workers", type=int,
                        help="The number of workers to use for data loading.", default=0)
    parser.add_argument("-s", "--shuffle", type=bool,
                        help="Shuffle the dataset before training.", default=True)
    parser.add_argument("-v", "--verbose", type=bool,
                        help="Enable verbose output.", default=False)
    parser.add_argument("-gpu", "--use_gpu", type=bool,
                        help="Enable training on gpu device.", default=True)
    parser.add_argument("-wb", "--wandb", type=boolean_type,
                        help="Enable logging with wandb.", default=True)
    
    # OUTPUT
    def list_type(s):
        return list(map(str, s.split(',')))
    parser.add_argument("-moddir", "--model_dir", type=str,
                        help="The folder to use for saving the model.", required=True)
    parser.add_argument("-wb", "--wandb", type=boolean_type,
                        help="Enable logging with wandb.", default=True)
    parser.add_argument("-wbnm", "--wb_name", type=str,
                        help="Name given for the wandb logger.", required=True)
    parser.add_argument("-wbno", "--wb_notes", type=str,
                        help="Notes given for the wandb logger.", required=True)
    parser.add_argument("-wbt", "--wb_tags", type=list_type,
                        help="List of tags for the wandb logger.", required=True)

    return parser.parse_args()

def training():
    # print banner and parse arguments
    banner()
    args = parse_args()

    # check if no path for loading the model is given
    if not args.load_model_dir:
        raise ValueError("No path to pretrained model given! Pretrained model is required for online training.")

    # setup wandb
    setup_wandb(args)

    # find device to train on
    device = find_best_device(args.use_gpu)

    # create replay buffer
    replay_buffer = ListReplayBuffer(
        max_seq_num=args.replay_buffer_size,
        batch_size=args.b,
    )

    # create transformer / load model
    # max sequence length has to be that of the dataset, which
    # was checked above
    action_space = args.act_space if args.act_enc else None
    action_dim = sum(action_space) if args.act_enc else args.act_dim
    model = OnlineDecisionTransformer(
        state_dim=args.state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        max_length=args.context_length,
        max_episode_length=args.max_ep_len,
        action_tanh=args.act_tanh,
        fancy_look_embedding=args.state_preprocessed,
        grid_size=args.grid_size,
        block_size=args.block_size,
        use_xformers=args.use_xformers,

    ).to(device)


    # setup optimizer, loss 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=args.step_size,
        gamma=args.lr_decay
    )

    log_temperature_optimizer = torch.optim.Adam(
            [model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )

    if args.loss_fn == "NLL":
        loss_fn = TenTenNLLActionLoss()
    elif args.loss_fn == "CE":
        loss_fn = TenTenCEActionLoss()
    elif args.loss_fn == "MSE":
        loss_fn = lambda y_hat, y: torch.mean((y_hat - y)**2)
    else:
        raise NotImplementedError(f"Unknown loss function {args.loss_fn}!")
    
    print("Waiting for Unity environment...")
    env = UnityEnvironment(randint(0, 2**16))
    env = UnityToGymWrapper(env)
    print("Unity environment started successfully! Starting training...")


    # training loop
    global_step = 0
    for epoch in tqdm(range(args.max_online_iterations), desc="Epochs", unit="epoch", leave=True, position=0):
        """
        1. Get new trajectory from environment (evaluate_episode_rtg)
        2. Add trajectory to replay buffer
        3. Sample batch from replay buffer and create dataloader from it
        4. Train model on batches from dataloader
            a. Get predictions from model
            b. Calculate loss
            c. Backpropagate loss
            d. Update model parameters
        5. Evaluate model on validation set (?)
        """
        
        epoch_loss = 0.0
        epoch_step = 0

        # Get one new trajectory from environment
        # makes sure it has the given minimum sequence length!
        trajectory = _collect_new_trajectory(
            env=env,
            state_dim=args.state_dim,
            action_dim=action_dim,
            model=model,
            device=device,
            target_return=40.,
            min_ep_len=args.min_seq_len,
            action_space=action_space
        )
        model.train()

        # Add trajectory to replay buffer
        replay_buffer.add(trajectory)

        # for I iterations
        
        # sample B trajectories according to sampling probalibity (implemented in replay buffer)
        # list[TensorDict]

        for tau_batch in tqdm(replay_buffer, desc="Batches", unit="batch", leave=True, position=1):
            # for each trajectory sample sub trajectory of length K
            actions = torch.zeros(args.bs, args.context_length, args.act_dim).to(device)
            states = torch.zeros(args.b, args.context_length, args.states_dim).to(device)
            rewards = torch.zeros(args.b, args.context_length, 1).to(device)
            timesteps = torch.linspace(0, args.context_length, args.context_length).to(device)
            attention_mask = torch.ones(args.context_length)

            for i, tau in enumerate(tau_batch):
                # get tau length from observation list
                tau_len = len(tau['observations'])
                # compute possible starting range by difference with sub trajectory length K
                max_idx = tau_len - args.context_length
                # sample random
                start_idx = randint(0, max_idx)
                end_idx = start_idx + args.context_length
                # fill subsequence into tensors
                actions[i] = tau['actions'][start_idx:end_idx]
                states[i] = tau['observations'][start_idx:end_idx]
                # replay buffer returns delta rewards, compute return to gos
                delta_rewards = tau['rewards'][start_idx:end_idx]
                cum_reward = 0.0
                for j in range(args.context_length-1, -1, -1):
                    cum_reward = delta_rewards[j]
                    rewards[j] = cum_reward

            # loss + SGD on sub trajectories to update N(mu,std) in online decision transformer
            if args.act_enc:
                actions = encode_actions(actions, action_space)
            action_targets = torch.clone(actions)
            _, _, action_preds = model(states, actions, rewards, timesteps, attention_mask)

            # use attention mask on prediction and targets to select
            action_preds = action_preds[attention_mask < 1]
            action_targets = action_targets[attention_mask < 1]
            
            # compute loss
            loss, log_likelihood, entropy = loss_fn(action_preds, action_targets, model.get_temperature().detach())
            epoch_loss += loss.item()
            epoch_step += 1

            log_temperature_optimizer.zero_grad()
            temperature_loss = (
                model.get_temperature() * (entropy - model.target_entropy).detach()
            )

            # backpropagate loss
            loss.backward()
            # clip gradient with value from paper github
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            temperature_loss.backward()
            log_temperature_optimizer.step()


        # log metrics
        epoch_loss /= epoch_step
        log_config = {
            "epoch": epoch,
            "epoch_step": epoch_step,
            "global_step": global_step,
            "train/epoch_loss": epoch_loss,
            "train/learning_rate": scheduler.get_last_lr()[0]
        }
        wandb.log(log_config, step=global_step)

        # save model after some epocds
        if epoch % args.save_every == 0:
            save_model(model, os.path.join(args.model_dir, f"model_e{epoch}.safetensors"))
        
        # let learning rate scheduler make a step
        scheduler.step()

    # save final model and optimizer
    save_model(model, os.path.join(args.model_dir, "model_final.safetensors"))

    # evaluation

    print("Start Evaluation:")

    # start evaluating loop
    with torch.no_grad():
        episodes_returns, episodes_lengths, _ = evaluate_episode_rtg(
            env,
            args.state_dim,
            action_dim,
            model,
            device,
            target_return=40.,
            nr_episodes=2,
            action_space=action_space,
            use_huggingface=args.hugging_transformer,
        )

    print(f"Average return and std: {episodes_returns.mean()},{episodes_returns.std()},\naverage length and std: {episodes_lengths.mean()},{episodes_lengths.std()}")

    wandb.log({
        "eval/ep_returns" : wandb.Histogram(episodes_returns.cpu()),
        "eval/ep_lengths" : wandb.Histogram(episodes_lengths.cpu())
    })


def _collect_new_trajectory(
        env,
        state_dim,
        action_dim,
        model,
        device,
        target_return,
        min_ep_len,
        action_space=(3, 10, 10)
    ):

    with torch.no_grad():
        nr_episodes = 1
        episodes_lengths = torch.zeros(nr_episodes).to(device)

        # ensure the trajectory has minimum length of min_ep_len
        # trajectory is a dictionary
        while episodes_lengths.sum() < min_ep_len:

            _, episodes_lengths, trajectory = evaluate_episode_rtg(
                env=env,
                state_dim=state_dim,
                action_dim=action_dim,
                model=model,
                device=device,
                target_return=target_return,
                nr_episodes=1,
                action_space=action_space,
                use_huggingface=False
            )
    
    # return as tensordict
    return TensorDict(trajectory)

if __name__=="__main__":
    training()
