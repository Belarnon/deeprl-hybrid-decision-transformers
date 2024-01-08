import torch
import sys

from trajectory_dataset import TrajectoryDataset

if False:
    raw_data = torch.load("Python\\dataset\\small.pt")
    if len(raw_data[-1]) == 0:
        raw_data = raw_data[:-1]
        torch.save(raw_data, "Python\\dataset\\small.pt")

#tds = TrajectoryDataset(5, 10, "Python\\dataset\\small.pt", needs_conversion=True)

tds = TrajectoryDataset(6, 10, "Python\\dataset\\small_converted.pt")

print("the end")