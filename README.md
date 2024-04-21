# Generative Adversarial Imitation Learning with PyTorch

This repository is for a simple implementation of Generative Adversarial Imitation Learning (GAIL) with PyTorch. This implementation is based on the original GAIL paper ([link](https://arxiv.org/abs/1606.03476)), and my Reinforcement Learning Collection repository ([link](https://github.com/hcnoh/rl-collection-pytorch)).

In this repository, [OpenAI Gym](https://gym.openai.com/) environments such as `CartPole-v0`, `Pendulum-v0`, and `BipedalWalker-v3` are used. You need to install them before running this repository.

*Note*: The environment's names could be different depending on the version of OpenAI Gym.

## Install Dependencies
1. Install Python 3.
2. Install the Python packages in `requirements.txt`. If you are using a virtual environment for Python package management, you can install all python packages needed by using the following bash command:

    ```bash
    $ pip install -r requirements.txt
    $ pip install torch
    $ pip install gymnasium[mujoco]
    ```

## Training and Running
1. Modify `config.json` as your machine setting.
2. Execute training process by `train.py`. An example of usage for `train.py` are following:

    ```bash
    $ python ensemble.py --env_name=Walker2d-v4 --num_bags=3 --num_layers=3
    ```

    The following bash command will help you:

    ```bash
    $ python train.py -h
    ```

## References
- The original GAIL paper: [link](https://arxiv.org/abs/1606.03476)
- Reinforcement Learning Collection with PyTorch: [link](https://github.com/hcnoh/rl-collection-pytorch)