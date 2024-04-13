import os
import json
import pickle
import argparse

import torch
import gymnasium as gym
import numpy as np
import statistics

from models.nets import Expert
from models.gail import GAIL
from utils.funcs import gather_expert_data, process_traj_data, bootstrap_expert_data

TRAJECTORY_ENVS = ["Walker2d-v4", "HalfCheetah-v4", "Hopper-v4", "Humanoid-v4", "HumanoidStandup-v4"]
ENVS = ["CartPole-v1", "Pendulum-v0", "BipedalWalker-v3"] + TRAJECTORY_ENVS

def bagging_train(env_name, num_bags=3):
    """Train models on env_name with different hidden size"""
    # load configs
    expert_ckpt_path = "experts"
    expert_ckpt_path = os.path.join(expert_ckpt_path, env_name)

    with open(os.path.join(expert_ckpt_path, "model_config.json")) as f:
        expert_config = json.load(f)
    
    ckpt_path = "ckpts"
    ckpt_path = os.path.join(ckpt_path, env_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open("config.json") as f:
        config = json.load(f)[env_name]

    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(config, f, indent=4)
        
    
    # make env
    env = gym.make(env_name)
    env.reset()
    
    state_dim = len(env.observation_space.high)
    if env_name in ["CartPole-v1"]:
        discrete = True
        action_dim = env.action_space.n
    else:
        discrete = False
        action_dim = env.action_space.shape[0]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # load experts
    if env_name in TRAJECTORY_ENVS:
        expert_data = np.load(os.path.join(expert_ckpt_path, "trajectories.npz"), allow_pickle=True)
        expert_data = process_traj_data(expert_data)
    else:
        expert = Expert(
            state_dim, action_dim, discrete, **expert_config
        ).to(device)
        expert.pi.load_state_dict(
            torch.load(
                os.path.join(expert_ckpt_path, "policy.ckpt"), map_location=device
            )
        )
        obs, acs, rews = gather_expert_data(env, expert, config["num_steps_per_iter"])
        expert_data = {"obs": obs, "acs": acs, "rews": rews}
        
    # split exper_data into num_bags
    bags = bootstrap_expert_data(expert_data, num_bags)
    
    # train model
    # TODO: add checkpoints for each model?
    models = []
    for i, data in enumerate(bags):
        print(f"training model {i+1}")
        new_model = GAIL(state_dim, action_dim, discrete, config, hidden_size=10).to(device)
        new_model.train(env, data, print_every=10)
        models.append(new_model)
    return models

def ensemble_act(state, models, is_discrete, weights=None):
    # get actions from each model
    acts = [model.act(state) for model in models]
    
    if is_discrete:
        return np.array([statistics.mode([val[0] for val in acts])])
        
    # assume average if weights is None
    if weights is None:
        num_models = len(models)
        weights = [1.0 / num_models for _ in range(num_models)]
    
    weighted_sum = sum(act * weight for act, weight in zip(acts, weights))
    total_weight = sum(weights)
    return weighted_sum / total_weight if total_weight != 0 else 0
        

def main(env_name):
    # load checkpoints
    ckpt_path = "ckpts"
    
    # check if checkpoint folder exists and env_name exists
    if not os.path.isdir(ckpt_path):
        print(f"No trained checkpoints on env {env_name}")
    
    if env_name not in ENVS:
        print("The environment name is wrong!")
        return

    # train models
    models = bagging_train(env_name)
    
    # evaluate
    # run 10 episodes with ensemble act
    num_ep = 10
    max_step = 1000
    
    env = gym.make(env_name)
    
    observation, info = env.reset()
    terminated = False
    
    ep_rewards = []
    for _ in range(num_ep):
        steps = 0
        ep_reward = 0
        while not terminated and not truncated and steps < max_step :
            action = ensemble_act(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
        ep_rewards.append(ep_reward)
    env.close()
    
    print(f"Reward Mean: {np.mean(ep_rewards)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="CartPole-v1",
        help=f"Type the environment name to run. \
            The possible environments are \
                {ENVS}"
    )
    args = parser.parse_args()

    main(**vars(args))
    