# Towards Hybrid Deep Reinforcement Learning with Decision Transformers for Large Discrete Action Spaces
This repository contains the code for the project "Towards Hybrid Deep Reinforcement Learning with Decision Transformers for Large Discrete Action Spaces" by Kevin Rohner, Philipp Brodmann and Bastian Schildknecht. The project was conducted as part of the course "Deep Learning" at ETH Zurich in the autumn semester 2023-2024.

## Table of Contents
* [Abstract](#abstract)
* [Project Structure](#project-structure)
* [Dependencies](#dependencies)
* [Project Setup](#project-setup)
* [Authors](#authors)
* [Citation](#citation)

## Abstract
In this study, we explore the use of offline and online decision transformers in the context of deep reinforcement learning. We apply a decision transformer to the problem of solving the game 1010!. We aim to compare the performance of the offline decision transformer to the online decision transformer.
We find that the unexpected high complexity and large action space of 1010! poses significant challenges to simpler decision transformer architectures and likely requires larger models and more training data to achieve good performance. Additionally, we explore the use of custom CNN-based observation encoders and find that they can accelerate training as shown in overfitting experiments.

## Project Structure
- `HybridDeepRL/` contains the Unity project for the game 1010! as an ML-Agents environment.
- `Python/` contains the Python code for training and evaluating the different decision transformer models.
- `Python/dataset/` contains the datasets and dataset implementations used for training and validation.
- `Python/evaluation/` contains the code for evaluating the trained models in the Unity environment.
- `Python/models/` contains pretrained models weights.
- `Python/modules/` contains code for reusable submodules used throughout the project.
- `Python/networks/` contains the code for the different decision transformer architectures.
- `Python/trainers/` contains the code for training and validation.
- `Python/utils/` contains utility code for various tasks like dataset handling, plotting, etc.
- `requirements.txt` contains the Python dependencies for the project.
- `README.md` is this file.
- `LICENSE` contains the license for this project.

## Dependencies
* Unity 2022.3.13f1
* Python 3.8.x

## Project Setup
1. Clone the repository.
2. Create a virtual python environment for python 3.8.x and activate it.
3. Install the python dependencies using `pip install -r requirements.txt`.
4. Open the Unity project `HybridDeepRL/` in Unity 2022.3.13f1.
5. Open a suitable scene from `HybridDeepRL/Assets/Scenes/`.
6. Run one of the preconfigured launch configurations in Visual Studio Code to train the offline or online decision transformer, respectively and press play in the Unity editor when prompted.

## Authors
* [Kevin Rohner](mailto:rohnerk@student.ethz.ch)
* [Philipp Brodmann](mailto:philipbr@student.ethz.ch)
* [Bastian Schildknecht](mailto:sbastian@student.ethz.ch)

## Citation
```
@misc{rohner2024hybrid,
  title={Towards Hybrid Deep Reinforcement Learning with Decision Transformers for Large Discrete Action Spaces},
  author={Rohner, Kevin and Brodmann, Philipp and Schildknecht, Bastian},
  journal={Unpublished},
  year={2021}
}
```
