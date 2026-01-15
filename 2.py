"""
经典强化学习 Demo：Q-learning（表格法）解决 CliffWalking（悬崖漫步）任务。

特点：
- 不依赖 gymnasium/gym，环境在本文件内手写，开箱即跑
- 算法使用“表格型”Q-learning：维护 Q(s,a) 表，而不是神经网络
- 训练后打印学到的策略，并保存回报曲线图 q_learning_returns.png

你可以把它当作强化学习最入门的“闭环”示例：
环境（Env） -> 智能体（Agent/Q 表） -> 采样交互（step） -> TD 更新（Q-learning） -> 评估/可视化
"""

import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class StepResult:
    """
    环境一步交互的返回结果（类似 gym 的 step 返回值简化版）。

    - state: 下一个状态（这里用 int 编号表示格子位置）
    - reward: 奖励
    - done: 回合是否结束（到达终点或掉下悬崖）
    """
    state: int
    reward: float
    done: bool


class CliffWalkingEnv:
    """
    CliffWalking（悬崖漫步）环境。

    网格默认 4x12：
    - 起点 S：左下角
    - 终点 G：右下角
    - 悬崖 X：底部一排中间的格子（掉下去大惩罚并结束回合）

    动作空间：4 个离散动作
    0=上, 1=右, 2=下, 3=左

    奖励设计（经典教材版）：
    - 普通走一步：-1
    - 到达终点：0（并 done=True）
    - 掉下悬崖：-100（并 done=True，且状态回到起点）
    """
    def __init__(self, height: int = 4, width: int = 12):
        self.height = height
        self.width = width
        self.start = (height - 1, 0)
        self.goal = (height - 1, width - 1)
        self.cliff = {(height - 1, c) for c in range(1, width - 1)}
        self._pos = self.start

    @property
    def n_states(self) -> int:
        # 状态数 = 网格总格子数，每个格子当作一个状态
        return self.height * self.width

    @property
    def n_actions(self) -> int:
        # 4 个方向动作
        return 4

    def reset(self) -> int:
        # 重置回合到起点，返回起点状态编号
        self._pos = self.start
        return self._state_id(self._pos)

    def step(self, action: int) -> StepResult:
        # 根据动作计算下一格位置（先按方向移动，再做边界裁剪）
        r, c = self._pos
        if action == 0:
            r -= 1
        elif action == 1:
            c += 1
        elif action == 2:
            r += 1
        elif action == 3:
            c -= 1
        else:
            raise ValueError(f"Invalid action: {action}")

        # 防止走出边界：把坐标裁剪到合法范围
        r = int(np.clip(r, 0, self.height - 1))
        c = int(np.clip(c, 0, self.width - 1))
        next_pos = (r, c)

        # 掉下悬崖：大惩罚，回合结束（并把状态设成起点，方便学习到“别走这里”）
        if next_pos in self.cliff:
            self._pos = next_pos
            return StepResult(state=self._state_id(self.start), reward=-100.0, done=True)

        # 到达终点：回合结束
        if next_pos == self.goal:
            self._pos = next_pos
            return StepResult(state=self._state_id(next_pos), reward=0.0, done=True)

        # 普通一步：-1，不结束
        self._pos = next_pos
        return StepResult(state=self._state_id(next_pos), reward=-1.0, done=False)

    def _state_id(self, pos: tuple[int, int]) -> int:
        # 把二维坐标 (row, col) 编码成一维状态编号：row * width + col
        r, c = pos
        return r * self.width + c

    def format_policy(self, q: np.ndarray) -> str:
        """
        把 Q 表的"贪心策略"打印成网格箭头图，方便直观看学到的策略。

        游戏规则说明：
        - 在4行×12列的网格中，从左下角起点S出发，目标是到达右下角终点G
        - 向右走至X区域（悬崖）会掉入悬崖，立刻返回起点并扣100分
        - 每走一步扣1分，到达终点G得0分并结束游戏
        - 箭头图展示每个网格位置学到的最优移动方向（上/右/下/左）

        符号说明：
        - S: 起点 (Start)
        - G: 终点 (Goal)  
        - X: 悬崖 (Cliff)
        - ↑→↓←: 其他格子的最优移动方向箭头
        """
        arrows = {0: "↑", 1: "→", 2: "↓", 3: "←"}
        lines: list[str] = []
        for r in range(self.height):
            row: list[str] = []
            for c in range(self.width):
                pos = (r, c)
                if pos == self.start:
                    row.append("S")
                elif pos == self.goal:
                    row.append("G")
                elif pos in self.cliff:
                    row.append("X")
                else:
                    s = self._state_id(pos)
                    a = int(np.argmax(q[s]))
                    row.append(arrows[a])
            lines.append(" ".join(row))
        return "\n".join(lines)


def epsilon_greedy_action(q: np.ndarray, state: int, epsilon: float, rng: random.Random) -> int:
    """
    ε-greedy（epsilon-greedy）选动作：
    - 以 epsilon 的概率随机探索（exploration）
    - 否则选择当前 Q 最大的动作（exploitation）
    """
    if rng.random() < epsilon:
        return rng.randrange(q.shape[1])
    return int(np.argmax(q[state]))


def q_learning(
    env: CliffWalkingEnv,
    episodes: int = 2000,
    alpha: float = 0.5,
    gamma: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    max_steps_per_episode: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, list[float]]:
    """
    Q-learning（离策略 off-policy 的时序差分方法）。

    我们维护一个 Q 表：q[state, action] 表示在 state 执行 action 的“长期收益估计”。

    更新公式（TD 更新）：
      Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

    参数含义（典型“超参数”）：
    - episodes：训练回合数（越多通常越好，但更慢）
    - alpha：学习率（更新幅度）
    - gamma：折扣因子（未来奖励重要程度）
    - epsilon_*：探索率的起止与衰减策略
    - max_steps_per_episode：单回合最多走多少步，防止死循环
    - seed：随机种子，保证复现
    """
    rng = random.Random(seed)
    # Q 表初始化为 0：形状 [状态数, 动作数]
    q = np.zeros((env.n_states, env.n_actions), dtype=np.float32)
    # 记录每个 episode 的总回报（用于画学习曲线）
    returns: list[float] = []

    epsilon = epsilon_start
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0

        for _step in range(max_steps_per_episode):
            action = epsilon_greedy_action(q, state, epsilon, rng)
            sr = env.step(action)

            # TD target = r + gamma * max_a' Q(s',a')（如果回合结束就没有下一步价值）
            td_target = sr.reward
            if not sr.done:
                td_target += gamma * float(np.max(q[sr.state]))
            # TD error = target - current
            td_error = td_target - float(q[state, action])
            # 按学习率 alpha 更新
            q[state, action] = float(q[state, action]) + alpha * td_error

            total_reward += sr.reward
            state = sr.state
            if sr.done:
                break

        returns.append(total_reward)
        # 训练过程中逐步降低探索率：越往后越“利用”学到的策略
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return q, returns


def evaluate_greedy_policy(
    env: CliffWalkingEnv,
    q: np.ndarray,
    episodes: int = 50,
    max_steps_per_episode: int = 1000,
) -> dict[str, float]:
    """
    评估：固定使用贪心策略（永远选 argmax Q），跑若干回合统计表现。

    返回：
    - avg_return：平均回报
    - success_rate：到达终点的成功率（%）
    """
    returns: list[float] = []
    successes = 0
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        for _step in range(max_steps_per_episode):
            # 贪心动作：不再探索
            action = int(np.argmax(q[state]))
            sr = env.step(action)
            total_reward += sr.reward
            state = sr.state
            if sr.done:
                # 在这个环境里：到达终点的 reward=0（掉悬崖 reward=-100）
                if sr.reward == 0.0:
                    successes += 1
                break
        returns.append(total_reward)
    return {
        "avg_return": float(np.mean(returns)),
        "success_rate": float(successes / episodes),
    }


def plot_returns(returns: list[float], out_path: str = "q_learning_returns.png") -> None:
    """
    绘制学习曲线（episode return）。
    为了更平滑，额外画一个滑动平均（moving average）。
    """
    window = 50
    series = np.array(returns, dtype=np.float32)
    if len(series) >= window:
        kernel = np.ones(window, dtype=np.float32) / window
        smooth = np.convolve(series, kernel, mode="valid")
        x = np.arange(len(smooth)) + window
        plt.figure(figsize=(10, 5))
        plt.plot(series, alpha=0.25, label="Return")
        plt.plot(x, smooth, label=f"Moving Avg ({window})")
    else:
        plt.figure(figsize=(10, 5))
        plt.plot(series, label="Return")

    plt.title("Q-learning on CliffWalking (Episode Return)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    # 1) 创建环境
    env = CliffWalkingEnv()
    # 2) 训练 Q-learning，得到 Q 表与训练曲线
    q, returns = q_learning(env)
    # 3) 用贪心策略评估（看最终学得怎样）
    metrics = evaluate_greedy_policy(env, q)
    # 4) 保存训练曲线图
    plot_returns(returns)

    print("Policy (S=start, G=goal, X=cliff):")
    print(env.format_policy(q))
    print("")
    print(f"Training episodes: {len(returns)}")
    print(f"Last 10 avg return: {float(np.mean(returns[-10:])):.2f}")
    print(f"Eval avg return: {metrics['avg_return']:.2f}")
    print(f"Eval success rate: {metrics['success_rate']*100:.1f}%")
    print("Saved plot: q_learning_returns.png")


if __name__ == "__main__":
    main()
