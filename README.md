# **Super Mario RL Project**

![Super Mario RL](demo/mario.gif)

## **Overview**

This project focuses on implementing reinforcement learning algorithms to train agents in the Super Mario Gym environment. The goal is to create an intelligent agent that can navigate and complete levels in the Super Mario environment using RL techniques dealing with the sparse environment problem.

## **Features**

- **Reinforcement Learning Algorithms**: Implement and experiment with various RL algorithms such as Sarsa, Double Deep Q-learning, A3C and ICM module.
- **Super Mario Gym Environment**: Utilize the OpenAI Gym environment for Super Mario, providing a simulation for training and evaluating the agent.
- **Visualization**: Include visualizations and graphs to demonstrate the learning progress of the agent over time.

## **Getting Started**

### **Prerequisites**

- Python 3
- All required packages are listed in the `requirements.txt` file.

### **Installation**

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

## **Executing**
You can run the program with various options
```bash
python3 main.py [OPTIONS]
```

**Command-Line Arguments**:
- '-t', '--train': Enable training mode.
- '-e', '--evaluate': Enable evaluation mode.
- 'algorithm <algorithm>': Specify the algorithm to use. Options are ddqn, ddqn_per, a2c. Default is a2c.
- '--episodes' <num_episodes>': Set the number of episodes:
    - Default for training is 20000;
- '--icm': Specify to use the ICM module. Default is False.
- ' --log-freq <interval>': Logging frequency (for A2C it is relative to the number of episodes, for DDQN is relative to the global step count). Default is 10.
- '--save-freq <interval>': Saving frequency (for A2C it is relative to the number of episodes, for DDQN is relative to the global step count). Default is 100.
- '--log-dir <path>': Directory to save logs. Default is ./logs/.
- '--save-dir <path>': Directory to save trained models. Default is ./trained_models/.
- '--model <model>': Specify if you want to load a specific model to continue the training or to evaluate.
- '--tb': Enable tensorboard.

### Examples

1. Run training with default settings
```bash
python3 main.py -t
```

2. Run training with a specific algorithm and number of episodes (and TensorBoard logging)
```bash
python3 main.py -t --episodes 25000 --algorithm a2c --icm --tb
```

3. Run evaluation

```bash
python3 main.py -e --algorithm a2c --icm --model mario_net.chkpt
```

## Win model
```bash
python3 main.py -e --algorithm ddqn --icm --model mario_net.chkpt
```
![Super Mario RL](demo/mario.gif)

## Experiments on Level 2
<div style="display: flex; justify-content: space-around; align-items: center;"> <img src="demo/lv2_go_up.gif" alt="Lv2 go up" style="width: 32%;"> <img src="demo/lv2_stuck.gif" alt="LV2 stuck" style="width: 32%;"> <img src="demo/lv2_sfida.gif" alt="Lv2 challenge" style="width: 32%;"> <img src="demo/lv2_good.gif" alt="Lv2 good" style="width: 32%;"> </div>





![Super Mario RL](demo/lv2_good.gif)

## Contacts

- **Davide Esposito Pelella:** [Davide Esposito Pelella](https://github.com/DavideEspositoPelella)
- **Paolo Ferretti:** [pabfr99](https://github.com/pabfr99)
