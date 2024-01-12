import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import os
import argparse
import pyfiglet
from safetensors.torch import save_file, load_file
from tqdm import tqdm

import transformers
from transformers.models.decision_transformer.modeling_decision_transformer import DecisionTransformerOutput

from dataset.trajectory_dataset import TrajectoryDataset
from networks.decision_transformer import DecisionTransformer
from utils.training_utils import find_best_device, encode_actions, decode_actions
from modules.loss.action_crossentropy import TenTenActionLoss

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
    print(pyfiglet.figlet_format("HDT", font="slant"))

def parse_args():
    """
    Parses the arguments for the pretraining gym.
    """

    parser = argparse.ArgumentParser(description="Trains the Hybrid Decision Transformer HDT")
    # DATASET
    parser.add_argument("-ds", "--dataset", type=str,
                        help="The dataset to use as training data.", required=True)
    parser.add_argument("-conv", "--conversion", type=bool,
                        help="Flag to indicate that the given dataset needs to be converted.", default=False)
    parser.add_argument("-minlen", "--min_seq_len", type=int,
                        help="The (inclusive) minimum length of sequences to train on.", required=True)
    parser.add_argument("-maxlen", "--max_seq_len", type=int,
                        help="The (inclusive) maximum length of sequences to train on.", required=True)
    

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
                        help="Set tanh layer at the end of action predictor.", default=False)
    
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
    parser.add_argument("-gpu", "--use_gpu", type=bool,
                        help="Enable training on gpu device.", default=True)
    
    # OUTPUT
    parser.add_argument("-moddir", "--model_dir", type=str,
                        help="The folder to use for saving the model.", required=True)
    parser.add_argument("-se", "--save_every", type=int,
                        help="Save the model every n epochs.", default=1)

    return parser.parse_args()

def training():
    # print banner and parse arguments
    banner()
    args = parse_args()

    # find device to train on
    device = find_best_device(args.use_gpu)

    # load dataset
    tds = TrajectoryDataset(
        args.min_seq_len,
        args.max_seq_len,
        args.dataset,
        args.conversion,
        device
    )

    # after loading the dataset, check for the maximum sequence length
    dataset_max_seq_len = tds.max_subseq_length

    # create dataloader
    dataloader = DataLoader(
        dataset=tds,
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
            args.state_dim,
            action_dim,
            args.hidden_dim,
            dataset_max_seq_len,
            args.max_ep_len,
            args.act_tanh
        ).to(device)


    if args.load_model:
        model.load_state_dict(load_file(args.load_model_dir))
        model.eval()

    # setup optimizer, loss 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=args.step_size,
        gamma=args.lr_decay
    )

    if args.loss_fn == "CE":
        loss_fn = TenTenActionLoss()
    elif args.loss_fn == "MSE":
        loss_fn = lambda y_hat, y: torch.mean((y_hat - y)**2)
    else:
        raise NotImplementedError(f"Unknown loss function {args.loss_fn}!")

    # start training loop
    step, epoch_loss = 0, 0

    epoch_progress = tqdm(range(args.epochs), desc="Epochs", unit="epoch", leave=True, position=0)
    for epoch in epoch_progress:
        for batch in tqdm(dataloader, desc="Batches", unit="batch", leave=False, position=1):

            # Get data
            states = batch[0].to(device).squeeze()
            actions = batch[1].to(device).squeeze()
            actions = encode_actions(actions, action_space) if args.act_enc else actions
            rtg = batch[2].to(device).unsqueeze(-1)
            timesteps = batch[3].to(device).squeeze()
            attention_mask = batch[4].to(device).squeeze()

            actions_target = torch.clone(actions)

            # reset gradients
            optimizer.zero_grad()

            # forward pass
            if args.hugging_transformer:
                output: DecisionTransformerOutput = model(states, actions, None, rtg, timesteps, attention_mask)
                action_preds = output.action_preds
            else:
                _, _, action_preds = model(states, actions, rtg, timesteps, attention_mask)

            # compute loss
            loss = loss_fn(action_preds, actions_target)
            epoch_loss += loss.item()
            epoch_progress.set_postfix({"loss": loss.item()})

            # backprop
            loss.backward()
            optimizer.step()
            step += 1

        # save model after some epocds
        if epoch % args.save_every == 0:
            save_file(model.state_dict(), os.path.join(args.model_dir, f"model_e{epoch}.safetensors"))
        
        # let learning rate scheduler make a step
        scheduler.step()

    # save final model and optimizer
    save_file(model.state_dict(), os.path.join(args.model_dir, "model_final.safetensors"))
    
if __name__=="__main__":
    training()
