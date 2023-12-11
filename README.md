# Super Mario RL Project

![Super Mario RL](images/mario.gif)

## Overview

This project focuses on implementing reinforcement learning algorithms to train agents in the Super Mario Gym environment. The goal is to create an intelligent agent that can navigate and complete levels in the Super Mario environment using RL techniques dealing with the sparse environment problem.

## Features

- **Reinforcement Learning Algorithms**: Implement and experiment with various RL algorithms such as Sarsa, Double Deep Q-learning, A3C.
- **Super Mario Gym Environment**: Utilize the OpenAI Gym environment for Super Mario, providing a realistic simulation for training and evaluating the agent.
- **Visualization**: Include visualizations and graphs to demonstrate the learning progress of the agent over time.

## Getting Started

### Prerequisites

- Python 3
- All required packages are listed in the `requirements.txt` file.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/DavideEspositoPelella/SuperMario-RL.git
```
2. Navigate the DDQN folder
```bash
cd SuperMario-RL/DDQN
```
3. Set up a Python Environment:

- Using a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate # On Windows use 'venv/Scripts/activate' 
```
- Using Conda 
```bash
conda create -n supermario_rl python=3.9
conda activate supermario_rl
```
3. Install dependencies.
```bash
pip install -r requirements.txt
```
4. Run training
```bash
python3 main.py -t -episodes <num_episodes>
```

## Contacts

- **Davide Esposito Pelella:** [Davide Esposito Pelella](https://github.com/DavideEspositoPelella)
- **Paolo Ferretti:** [pabfr99](https://github.com/pabfr99)
