import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import os
import argparse
import pyfiglet
from safetensors.torch import save_file, load_file
from tqdm import tqdm


from dataset.trajectory_dataset import TrajectoryDataset
from networks.decision_transformer import DecisionTransformer
from utils.setup_methods import find_best_device

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
                        help="The learning rate to use for training.", default=0.001)
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

def encode_actions(action_batch: torch.tensor, action_space=(3,10,10)):
    """
    Encodes actions into one-hot vectors.

    action_batch: batch of action sequences
    action_space: tuple of action space dimensions
    """
    
    encoded_actions = torch.zeros(action_batch.shape[0], action_batch.shape[1], sum(action_space), device=action_batch.device)

    offset = [0] + list(action_space)[:-1]
    for i, sequence in enumerate(action_batch):
        for j, action in enumerate(sequence):
            for k, a in enumerate(action):
                encoded_actions[i, j, offset[k] + int(a)] = 1

    return encoded_actions

def decode_actions(action_batch: torch.tensor, action_space=(3,10,10)):
    """
    Decodes one-hot action vectors into action sequences.

    actions_batch: batch of action sequences
    action_space: tuple of action space dimensions
    """

    decoded_actions = torch.zeros(action_batch.shape[0], action_batch.shape[1], len(action_space), device=action_batch.device)

    limits = np.insert(np.cumsum(list(action_space)), 0,0)
    for i, sequence in enumerate(action_batch):
        for j, action in enumerate(sequence):
            
            act = torch.zeros(len(action_space))
            for k in range(len(action_space)):
                act[k] = action[limits[k]:limits[k+1]].argmax()

            decoded_actions[i, j, :] = act

    return decoded_actions

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

    # create dataloader
    dataloader = DataLoader(
        dataset=tds,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.workers
    )

    # create transformer / load model
    action_space = args.act_space if args.act_enc else None
    action_dim = sum(action_space) if args.act_enc else args.act_dim
    model = DecisionTransformer(
        args.state_dim,
        action_dim,
        args.hidden_dim,
        args.max_seq_len,
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

    # TODO how to define loss function?
    if args.loss_fn == "CE":
        loss_fn = torch.nn.CrossEntropyLoss()
    elif args.loss_fn == "MSE":
        loss_fn = lambda y_hat, y: torch.mean((y_hat - y)**2)
    else:
        raise NotImplementedError(f"Unknown loss function {args.loss_fn}!")

    # start training loop
    step, epoch_loss = 0, 0

    for epoch in tqdm(range(args.epochs), desc="Epochs", unit="epoch", leave=True, position=0):
        for batch in tqdm(dataloader, desc="Batches", unit="batch", leave=False, position=1):

            # Get data
            states = batch[0].to(device).squeeze()
            actions = batch[1].to(device).squeeze()
            actions = encode_actions(actions, action_space) if args.act_enc else actions
            rtg = batch[2].to(device).unsqueeze(-1)
            timesteps = batch[3].to(device).squeeze()
            attention_mask = batch[4].to(device).squeeze()

            # reset gradients
            optimizer.zero_grad()

            # forward pass
            return_preds, state_preds, action_preds = model(states, actions, rtg, timesteps, attention_mask)

            # compute loss
            loss = loss_fn(action_preds) # TODO: Define target for loss function xD, so it can be called like `loss_fn(y_hat, y)`
            epoch_loss += loss.item()

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
    save_file(model.state_dict(), os.path.join(args.model_dir, f"model_final.safetensors"))
    
if __name__=="__main__":
    training()
