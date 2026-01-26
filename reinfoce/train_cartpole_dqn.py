import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import gymnasium as gym
except ImportError:
    import gym


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x):
        return self.net(x)


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
):
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = QNetwork(obs_dim, act_dim).to(device)
    target_q = QNetwork(obs_dim, act_dim).to(device)
    target_q.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=1e-3)

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = model_path if model_path else os.path.join(models_dir, "cartpole_dqn.pth")

    if os.path.exists(model_path) and not force_train:
        q.load_state_dict(torch.load(model_path, map_location=device))
    else:
        buffer = ReplayBuffer(100000, obs_dim)
        gamma = 0.99
        batch_size = 64
        start_steps = 1000
        target_update = 1000
        eps_start = 1.0
        eps_end = 0.05
        eps_decay = 30000

        steps = 0
        episode = 0
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
                ep_reward = 0.0

            if steps >= start_steps:
                o_b, a_b, r_b, no_b, d_b = buffer.sample(batch_size)
                o_b = o_b.to(device)
                a_b = a_b.to(device)
                r_b = r_b.to(device)
                no_b = no_b.to(device)
                d_b = d_b.to(device)

                q_vals = q(o_b).gather(1, a_b.view(-1, 1)).squeeze(1)
                with torch.no_grad():
                    next_q = target_q(no_b).max(1).values
                    target = r_b + (1.0 - d_b) * gamma * next_q

                loss = nn.MSELoss()(q_vals, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (steps + 1) % target_update == 0:
                target_q.load_state_dict(q.state_dict())

            steps += 1

        torch.save(q.state_dict(), model_path)

    returns = []
    for _ in range(10):
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
    print(f"AvgEvalReturn={avg_ret:.2f}")
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
    args = parser.parse_args()
    train(
        render=args.render,
        total_steps=args.total_steps,
        model_path=args.model_path,
        force_train=args.force_train,
        render_episodes=args.render_episodes,
        render_delay=args.render_delay,
    )
