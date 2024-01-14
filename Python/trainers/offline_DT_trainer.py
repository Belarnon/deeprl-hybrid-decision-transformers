import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader, random_split

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
from networks.decision_transformer import DecisionTransformer
from utils.training_utils import find_best_device, encode_actions, decode_actions, setup_wandb

"""
Technically this is not a gym, as it does not use Unity ML Agents.
This class provides the complete offline pretraining of a decision transformer, including:
    - argument parsing
    - loading/creating dataset
    - training
    - export model

code is heavily inspired by https://github.com/bastianschildknecht/cil-ethz-2023-kebara/blob/main/train_latent_mlp.py
    
"""

def banner():
    print(pyfiglet.figlet_format("HDT-Offline Trainer", font="slant"))

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
    # DATASET
    parser.add_argument("-ds", "--dataset", type=str,
                        help="The dataset to use as training data.", required=True)
    parser.add_argument("-dsval", "--dataset_validation", type=str,
                        help="The dataset to use as validation data.", default=None)
    parser.add_argument("-conv", "--conversion", type=bool,
                        help="Flag to indicate that the given dataset needs to be converted.", default=False)
    parser.add_argument("-minlen", "--min_seq_len", type=int,
                        help="The (inclusive) minimum length of sequences to train on.", required=True)
    parser.add_argument("-maxlen", "--max_seq_len", type=int,
                        help="The (inclusive) maximum length of sequences to train on.", required=True)
    parser.add_argument("-stride", "--stride", type=int,
                        help="The stride to use for the sliding window.", default=1)
    parser.add_argument("-valsplit", "--validation_split", type=float,
                        help="The percentage of the dataset to use for validation.", default=0.2)
    

    # TRANSFORMER
    def tuple_type(s):
        return tuple(map(int, s.split(',')))
    parser.add_argument("-hugtrans", "--hugging_transformer", type=bool,
                        help="Flag to indicate whether the huggingface transformer should be used.", default=False)
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
    
    # TRAINING
    #   LOAD AND SAVE
    parser.add_argument("-lm", "--load_model", type=bool,
                        help="Load the model", default=False)
    parser.add_argument("-lmd", "--load_model_dir", type=str,
                        help="The filepath to the model that should be loaded.", default="")
    
    #   TRAINING ARGUMENTS
    parser.add_argument("-b", "--batch_size", type=int,
                        help="The batch size to use for training.", default=16)
    parser.add_argument("-e", "--epochs", type=int,
                        help="The number of epochs to train.", default=2)
    parser.add_argument("-lr", "--learning_rate", type=float,
                        help="The learning rate to use for training.", default=1e-4)
    parser.add_argument("-ss", "--step_size", type=int,
                        help="The stepsize used in the SGD algorithm. ", default=1)
    parser.add_argument("-dc", "--lr_decay", type=float,
                        help="Decay of the learning rate for a Learning Rate Scheduler.", default=1.0)
    parser.add_argument("-lfn", "--loss_fn", type=str,
                        help="Name of loss function to use.", default="CE")

    #   MISCELLANEOUS
    parser.add_argument("-w", "--workers", type=int,
                        help="The number of workers to use for data loading.", default=0)
    parser.add_argument("-s", "--shuffle", type=bool,
                        help="Shuffle the dataset before training.", default=True)
    parser.add_argument("-v", "--verbose", type=bool,
                        help="Enable verbose output.", default=False)
    parser.add_argument("-gpu", "--use_gpu", type=boolean_type,
                        help="Enable training on gpu device.", default=True)
    
    
    # OUTPUT
    def list_type(s):
        return list(map(str, s.split(',')))
    parser.add_argument("-moddir", "--model_dir", type=str,
                        help="The folder to use for saving the model.", required=True)
    parser.add_argument("-se", "--save_every", type=int,
                        help="Save the model every n epochs.", default=1)
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

    # setup wandb
    setup_wandb(args)

    # find device to train on
    device = find_best_device(args.use_gpu)

    # load dataset
    trajectory_train = TrajectoryDataset(
        args.min_seq_len,
        args.max_seq_len,
        args.stride,
        args.dataset
    )

    # after loading the dataset, check for the maximum sequence length
    dataset_max_seq_len = trajectory_train.max_subseq_length

    if args.dataset_validation is not None:
        trajectory_val = TrajectoryDataset(
            args.min_seq_len,
            args.max_seq_len,
            args.stride,
            args.dataset_validation
        )
        assert dataset_max_seq_len == trajectory_val.max_subseq_length, "The maximum sequence length of the validation dataset does not match the training dataset!"
    else:
        # create train & val random split
        dataset_lengths = [int(len(trajectory_train) * (1 - args.validation_split)), int(len(trajectory_train) * args.validation_split)]
        trajectory_train, trajectory_val = random_split(trajectory_train, dataset_lengths)



    # create dataloaders
    dataloader_train = DataLoader(
        dataset=trajectory_train,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.workers
    )
    dataloader_val = DataLoader(
        dataset=trajectory_val,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.workers
    )

    # create transformer / load model
    # max sequence length has to be that of the dataset, which
    # was checked above
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
        model = transformers.DecisionTransformerModel(config).to(device)
    else:
        model = DecisionTransformer(
            state_dim=args.state_dim,
            action_dim=action_dim,
            hidden_dim=args.hidden_dim,
            max_length=dataset_max_seq_len,
            max_episode_length=args.max_ep_len,
            action_tanh=args.act_tanh,
            fancy_look_embedding=args.state_preprocessed,
            grid_size=args.grid_size,
            block_size=args.block_size
        ).to(device)


    if args.load_model:
        model.load_state_dict(load_model(args.load_model_dir))
        model.eval()

    # setup optimizer, loss 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=args.step_size,
        gamma=args.lr_decay
    )

    if args.loss_fn == "CE":
        loss_fn = TenTenCEActionLoss()
    elif args.loss_fn == "NLL":
        loss_fn = TenTenNLLActionLoss()
    elif args.loss_fn == "MSE":
        loss_fn = lambda y_hat, y: torch.mean((y_hat - y)**2)
    else:
        raise NotImplementedError(f"Unknown loss function {args.loss_fn}!")

    # load the environment
    print("Waiting for Unity environment...")
    env = UnityEnvironment(seed=randint(0, 2**16))
    env = UnityToGymWrapper(env, allow_multiple_obs=True)
    print("Unity environment started successfully! Starting training...")

    # start training loop
    global_step = 0
    epoch_progress = tqdm(range(args.epochs), desc="Epochs", unit="epoch", leave=True, position=0)
    for epoch in epoch_progress:
        epoch_loss = 0
        epoch_step = 0
        for batch in tqdm(dataloader_train, desc="Batches", unit="batch", leave=False, position=1):

            # Get data
            states = batch[0].to(device)
            actions = batch[1].to(device)
            actions = encode_actions(actions, action_space) if args.act_enc else actions
            rtg = batch[2].to(device)
            timesteps = batch[3].to(device)
            attention_mask = batch[4].to(device)

            actions_target = torch.clone(actions)

            # reset gradients
            optimizer.zero_grad()

            # forward pass
            if args.hugging_transformer:
                output: DecisionTransformerOutput = model(states, actions, None, rtg, timesteps, attention_mask)
                action_preds = output.action_preds
            else:
                _, _, action_preds = model(states, actions, rtg, timesteps, attention_mask)

            # use attention mask on prediction and targets to select
            action_preds = action_preds[attention_mask < 1]
            actions_target = actions_target[attention_mask < 1]

            # compute loss
            loss = loss_fn(action_preds, actions_target)
            epoch_loss += loss.item()
            epoch_progress.set_postfix({"loss": loss.item()})

            # backprop
            loss.backward()
            # clip gradient with value from paper github
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            epoch_step += 1

        global_step += epoch_step

        # validation loss
        val_loss = 0
        val_step = 0
        with torch.no_grad():
            validation_progress = tqdm(dataloader_val, desc="Validation", unit="batch", leave=False, position=1)
            for batch in validation_progress:
                # Get data
                states = batch[0].to(device)
                actions = batch[1].to(device)
                actions = encode_actions(actions, action_space) if args.act_enc else actions
                rtg = batch[2].to(device)
                timesteps = batch[3].to(device)
                attention_mask = batch[4].to(device)

                actions_target = torch.clone(actions)

                # forward pass
                if args.hugging_transformer:
                    output: DecisionTransformerOutput = model(states, actions, None, rtg, timesteps, attention_mask)
                    action_preds = output.action_preds
                else:
                    _, _, action_preds = model(states, actions, rtg, timesteps, attention_mask)

                # use attention mask on prediction and targets to select
                action_preds = action_preds[attention_mask < 1]
                actions_target = actions_target[attention_mask < 1]

                # compute loss
                loss = loss_fn(action_preds, actions_target)
                val_loss += loss.item()
                val_step += 1

                validation_progress.set_postfix({"loss": loss.item()})

        # log metrics
        epoch_loss /= epoch_step
        val_loss /= val_step
        log_config = {
            "epoch": epoch,
            "epoch_step": epoch_step,
            "global_step": global_step,
            "train/epoch_loss": epoch_loss,
            "train/learning_rate": scheduler.get_last_lr()[0],
            "val/epoch_loss": val_loss
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
        episodes_returns, episodes_lengths = evaluate_episode_rtg(
            env,
            args.state_dim,
            action_dim,
            model,
            device,
            target_return=40.,
            nr_episodes=10,
            action_space=action_space,
            use_huggingface=args.hugging_transformer,
        )

    print(f"Average return and std: {episodes_returns.mean()},{episodes_returns.std()},\naverage length and std: {episodes_lengths.mean()},{episodes_lengths.std()}")

    wandb.log({
        "eval/ep_returns" : wandb.Histogram(episodes_returns.cpu()),
        "eval/ep_lengths" : wandb.Histogram(episodes_lengths.cpu())
    })

if __name__=="__main__":
    training()
