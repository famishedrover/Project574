# RL project for CSE 574: Planning and Learning in AI

Using [`gym-minigrid`](https://github.com/maximecb/gym-minigrid) environments and RL implementation from [`torch-rl`](https://github.com/lcswillems/rl-starter-files).

## Commands to install

```
pip3 install -r requirements.txt
```

Install the packaged torch-ac (In the repo instead of the original torch-ac).
```
cd torch-ac
pip3 install .
```


To install RL algorithms in [`torch-ac`](https://github.com/lcswillems/torch-ac):
```
git clone https://github.com/lcswillems/torch-ac.git
cd torch-ac
pip3 install -e .
```

## Commands to run
```
python3 -m torch_rl.scripts.train --algo ppo --env MiniGrid-Empty-8x8-v0 --model GridWorld --save-interval 100 --frames 1000000
tensorboard --logdir storage/GridWorld
python3 -m torch_rl.scripts.visualize --env MiniGrid-Empty-8x8-v0 --model GridWorld
python3 -m torch_rl.scripts.evaluate --env MiniGrid-Empty-8x8-v0 --model GridWorld

```


To select a gpu use flag --gpuidx <gpu-value>
For example, 

```
python3 -m torch_rl.scripts.train --algo ppo --env MiniGrid-Empty-8x8-v0 --model GridWorld --save-interval 100 --frames 1000000 --gpuidx 7
```