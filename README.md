# Taxi-v3 Reinforcement Learning
A reinforcement learning project in OpenAI's Gym Taxi-v3 environment that trains an AI agent to pick up and drop off passengers using Q-learning while learning the optimal routes through exploration and rewards.

Built using Python3, Gymnasium, Numpy, Matplotlib and tqdm.

# Environment Setup
Environment: OpenAI Gymnasium Taxi-v3

[See documentation here](https://gymnasium.farama.org/environments/toy_text/taxi)

Algorithm: Q-Learning (RL Model Free Learning)

Objective: Optimal route selection to transport passengers with the least number of actions

## Clone the Repository
```bash
git clone https://github.com/peter-z3ng/csci323-taxi-v3.git
cd csci323-taxi-v3
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

gymnasium:  for the Taxi-v3 environment

numpy:  for Q-table and numerical operations

matplotlib:  for plotting training results

tqdm:  for progress bar during training

time:  for timing display

## Running the Agent
Once dependencies are installed, start training the taxi agent by running:

For macOS/ Linux
```bash
python3 taxi_v3.py
```

For Windows
```bash
python taxi-v3.py
```
Source code [here](./ft37.py)

## Dataset (Model-free RL)
-  5x5 grid world
-  4 fixed pick up/drop off points
-  6 possible actions (north, east, south, west, pick-up, drop-off)
-  500 possible states
-  Q-table size = 500 x 6 = 3000

## Reproducing Key Results
1. Train the model for 30000 episodes
2. Track rewards per episode, steps, average q-values, epsilon values
3. Plot learning curve
4. Observe the performance and optimal routes selection

## Authors
**Group name:** FT37

**Members:**
- Aw Qixuan Charlotte
- Hein Thura Min
- Jun Peng Brandon Toh
- Nan Phyu Sin Maung
- Paing Thit Xan
- Saung Hnin Phyu
