# RL project for CSE 574: Planning and Learning in AI

Using [`gym-minigrid`](https://github.com/maximecb/gym-minigrid) environments and PPO from [`torch-ac`](https://github.com/lcswillems/torch-ac) RL implementation.

## Commands to run

```
pip3 install -r requirements.txt
python3 -m scripts.train --algo ppo --env MiniGrid-Empty-8x8-v0 --model GridWorld --save-interval 100 --frames 1000000
tensorboard --logdir storage/GridWorld
python3 -m scripts.visualize --algo ppo --env MiniGrid-Empty-8x8-v0 --model GridWorld
python3 -m scripts.evaluate --algo ppo --env MiniGrid-Empty-8x8-v0 --model GridWorld

```
