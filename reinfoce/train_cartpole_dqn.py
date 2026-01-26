import os
import time
import math
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import gymnasium as gym
except ImportError:
    import gym


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, dueling=True):
        super().__init__()
        hidden_sizes = [256, 256, 128]
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.feature = nn.Sequential(*layers)
        self.dueling = dueling
        if dueling:
            self.value_head = nn.Linear(in_dim, 1)
            self.adv_head = nn.Linear(in_dim, act_dim)
        else:
            self.q_head = nn.Linear(in_dim, act_dim)

    def forward(self, x):
        feat = self.feature(x)
        if self.dueling:
            value = self.value_head(feat)
            adv = self.adv_head(feat)
            return value + adv - adv.mean(dim=1, keepdim=True)
        return self.q_head(feat)


class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.full = False

    def add(self, o, a, r, no, d):
        self.obs[self.idx] = o
        self.actions[self.idx] = a
        self.rewards[self.idx] = r
        self.next_obs[self.idx] = no
        self.dones[self.idx] = d
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size):
        max_idx = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, max_idx, size=batch_size)
        o = torch.from_numpy(self.obs[idxs])
        a = torch.from_numpy(self.actions[idxs])
        r = torch.from_numpy(self.rewards[idxs])
        no = torch.from_numpy(self.next_obs[idxs])
        d = torch.from_numpy(self.dones[idxs])
        return o, a, r, no, d


def reset_env(env):
    result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        return result[0]
    return result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, _ = result
        done = terminated or truncated
        return obs, reward, done
    obs, reward, done, _ = result
    return obs, reward, done


def train(
    render=False,
    total_steps=100000,
    model_path="",
    force_train=False,
    render_episodes=5,
    render_delay=1 / 60,
    fast=False,
    fast_plus=False,
    lr=1e-3,
    buffer_size=100000,
    batch_size=64,
    start_steps=1000,
    target_update=1000,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=30000,
    stop_avg_return=450,
    stop_window=20,
    min_episodes=20,
    double_dqn=True,
    dueling=True,
    eval_episodes=10,
):
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = QNetwork(obs_dim, act_dim, dueling=dueling).to(device)
    target_q = QNetwork(obs_dim, act_dim, dueling=dueling).to(device)
    target_q.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=lr)

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = model_path if model_path else os.path.join(models_dir, "cartpole_dqn.pth")

    loaded = False
    if os.path.exists(model_path) and not force_train:
        try:
            q.load_state_dict(torch.load(model_path, map_location=device))
            loaded = True
        except RuntimeError:
            loaded = False

    trained_steps = 0
    trained_episodes = 0
    trained_seconds = 0.0

    if not loaded:
        start_time = time.time()
        if fast or fast_plus:
            total_steps = min(total_steps, 120000 if fast_plus else 80000)
            lr = 2.0e-3 if fast_plus else 2.5e-3
            buffer_size = 80000 if fast_plus else 60000
            batch_size = 256 if fast_plus else 128
            start_steps = 500 if fast_plus else 300
            target_update = 1000 if fast_plus else 700
            gamma = 0.995 if fast_plus else 0.99
            eps_decay = 15000 if fast_plus else 10000
            optimizer = optim.Adam(q.parameters(), lr=lr)

        buffer = ReplayBuffer(buffer_size, obs_dim)

        steps = 0
        episode = 0
        returns_window = deque(maxlen=stop_window)
        o = reset_env(env)
        ep_reward = 0.0

        while steps < total_steps:
            epsilon = eps_end + (eps_start - eps_end) * math.exp(-1.0 * steps / eps_decay)
            if random.random() < epsilon:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    qv = q(torch.from_numpy(o).float().unsqueeze(0).to(device))
                    a = int(torch.argmax(qv, dim=1).item())

            no, r, done = step_env(env, a)
            buffer.add(o, a, r, no, float(done))
            ep_reward += r
            o = no

            if done:
                o = reset_env(env)
                episode += 1
                returns_window.append(ep_reward)
                ep_reward = 0.0
                if (
                    episode >= min_episodes
                    and len(returns_window) == stop_window
                    and (sum(returns_window) / stop_window) >= stop_avg_return
                ):
                    break

            if steps >= start_steps:
                o_b, a_b, r_b, no_b, d_b = buffer.sample(batch_size)
                o_b = o_b.to(device)
                a_b = a_b.to(device)
                r_b = r_b.to(device)
                no_b = no_b.to(device)
                d_b = d_b.to(device)

                q_vals = q(o_b).gather(1, a_b.view(-1, 1)).squeeze(1)
                with torch.no_grad():
                    if double_dqn:
                        next_actions = q(no_b).argmax(1)
                        next_q = target_q(no_b).gather(1, next_actions.view(-1, 1)).squeeze(1)
                    else:
                        next_q = target_q(no_b).max(1).values
                    target = r_b + (1.0 - d_b) * gamma * next_q

                loss = nn.MSELoss()(q_vals, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (steps + 1) % target_update == 0:
                target_q.load_state_dict(q.state_dict())

            steps += 1

        trained_steps = steps
        trained_episodes = episode
        trained_seconds = time.time() - start_time

        torch.save(q.state_dict(), model_path)

    returns = []
    for _ in range(eval_episodes):
        o = reset_env(env)
        ret = 0.0
        done = False
        while not done:
            with torch.no_grad():
                qv = q(torch.from_numpy(o).float().unsqueeze(0).to(device))
                a = int(torch.argmax(qv, dim=1).item())
            o, r, done = step_env(env, a)
            ret += r
        returns.append(ret)
    avg_ret = sum(returns) / len(returns)
    if loaded:
        print("Mode=Inference (LoadedModel)")
    else:
        print("Mode=Training")
        print(f"TrainedSteps={trained_steps}")
        print(f"TrainedEpisodes={trained_episodes}")
        print(f"TrainSeconds={trained_seconds:.2f}")
    print(f"AvgEvalReturn={avg_ret:.2f}")
    print(f"EvalEpisodes={eval_episodes}")
    print(f"ModelSaved={model_path}")

    if render:
        render_env = gym.make("CartPole-v1", render_mode="human")
        for _ in range(render_episodes):
            o = reset_env(render_env)
            done = False
            while not done:
                with torch.no_grad():
                    qv = q(torch.from_numpy(o).float().unsqueeze(0).to(device))
                    a = int(torch.argmax(qv, dim=1).item())
                o, _, done = step_env(render_env, a)
                time.sleep(render_delay)
        render_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--total-steps", type=int, default=100000)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--render-episodes", type=int, default=5)
    parser.add_argument("--render-delay", type=float, default=1 / 60)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--fast-plus", action="store_true")
    parser.add_argument("--stop-avg-return", type=float, default=450)
    parser.add_argument("--stop-window", type=int, default=20)
    parser.add_argument("--no-double-dqn", action="store_true")
    parser.add_argument("--no-dueling", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()
    train(
        render=args.render,
        total_steps=args.total_steps,
        model_path=args.model_path,
        force_train=args.force_train,
        render_episodes=args.render_episodes,
        render_delay=args.render_delay,
        fast=args.fast,
        fast_plus=args.fast_plus,
        stop_avg_return=args.stop_avg_return,
        stop_window=args.stop_window,
        double_dqn=not args.no_double_dqn,
        dueling=not args.no_dueling,
        eval_episodes=args.eval_episodes,
    )
