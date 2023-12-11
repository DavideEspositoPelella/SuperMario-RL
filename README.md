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

1. **Clone the repository**:
```bash
git clone https://github.com/DavideEspositoPelella/SuperMario-RL.git
```
2. **Navigate the SuperMario-RL folder**
```bash
cd SuperMario-RL
```
3. **Install tkinter** (required for certain graphical operations in Python):
```bash
sudo apt-get install python3.8-tk
```
4. **Set up a Python Environment**:

- Using a virtual environment (optional but recommended)
```bash
sudo apt install python3.8-venv
python3 -m venv venv
source venv/bin/activate # On Windows use 'venv/Scripts/activate' 
```
- Using Conda 
```bash
conda create -n supermario_rl python=3.8
conda activate supermario_rl
```
5. **Install dependencies**.
```bash
pip install --no-cache-dir -r requirements.txt
```
6. **Navigate the model folder** (available 'DDQN', 'A3C', #TODO)
```bash
cd DDQN
```
7. **Run training**
```bash
python3 main.py -t --episodes <num_episodes>
```

## Contacts

- **Davide Esposito Pelella:** [Davide Esposito Pelella](https://github.com/DavideEspositoPelella)
- **Paolo Ferretti:** [pabfr99](https://github.com/pabfr99)
