# 关于银行执行股票交易策略的问题
# 我们可以将每个交易日看作一个环境状态，银行需要决定是否购买股票以及购买的数量。银行的目标是最大化利益
# 通过Q-learning进行训练，通过与环境交互来更新策略和价值函数，以找到最优的每日执行策略。
# 每个交易日作为一个状态，用于表示银行的决策点。
# 每个状态下银行可以采取的行动，包括购买股票的数量或者不购买。
# 奖励函数用于评估银行每个决策的好坏，可以根据利润或效用来定义。

import numpy as np
import gym


class StockTradingEnvironment(gym.Env):
    def __init__(self, num_shares=1000000, num_days=132, initial_price=100, volatility=0.4, discount_factor=0.99):
        """
        初始化股票交易环境

        参数:
            num_shares (int): 初始股票数量
            num_days (int): 总交易天数
            initial_price (float): 初始股票价格
            volatility (float): 价格波动率
            discount_factor (float): 折扣因子
        """
        self.num_shares = num_shares  # 初始股票数量
        self.num_days = num_days  # 总交易天数
        self.initial_price = initial_price  # 初始股票价格
        self.volatility = volatility  # 价格波动率
        self.discount_factor = discount_factor  # 折扣因子
        self.current_day = 0  # 当前交易天数
        self.current_price = initial_price  # 当前股票价格
        self.done = False  # 是否结束交易

    def step(self, action):
        """
        执行一个动作并返回新状态、奖励和是否结束的标志

        参数:
            action (int): 动作 (0表示持有股票，1表示买入股票)

        返回:
            state (int): 新状态 (当前交易天数)
            reward (float): 奖励
            done (bool): 是否结束交易
            info (dict): 其他信息 (空字典)
        """
        if self.done:
            raise ValueError("Episode is done. Please reset the environment.")

        if self.current_day >= self.num_days:
            self.done = True
            return self._get_state(), 0, self.done, {}

        if action == 1:  # 买入
            self.num_shares -= 1
        self.current_day += 1
        self.current_price *= np.exp(self.volatility * np.random.normal(0, 1))
        reward = (self.current_price - self.initial_price) * self.num_shares - action * self.current_price
        return self._get_state(), reward, self.done, {}

    def reset(self):
        """
        重置环境状态并返回初始状态

        返回:
            state (int): 初始状态 (当前交易天数)
        """
        self.current_day = 0
        self.current_price = self.initial_price
        self.done = False
        return self._get_state()

    def _get_state(self):
        """
        获取当前状态

        返回:
            state (int): 当前状态 (当前交易天数)
        """
        return self.current_day

    def render(self):
        """
        打印当前交易信息

        """
        print("交易天数:", self.current_day)
        print("股票价格:", self.current_price)
        print("持有的股票数量:", self.num_shares)


class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.99, exploration_rate=0.1):
        """
        初始化QLearning强化学习算法

        参数:
            num_states (int): 状态数量
            num_actions (int): 动作数量
            learning_rate (float): 学习率
            discount_factor (float): 折扣因子
            exploration_rate (float): 探索率
        """
        self.num_states = num_states  # 状态数量
        self.num_actions = num_actions  # 动作数量
        self.learning_rate = learning_rate  # 学习率
        self.discount_factor = discount_factor  # 折扣因子
        self.exploration_rate = exploration_rate  # 探索率
        self.q_table = np.zeros((num_states, num_actions))  # Q值表

    def select_action(self, state):
        """
        根据当前状态选择动作

        参数:
            state (int): 当前状态

        返回:
            action (int): 选择的动作
        """
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.num_actions)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        """
        更新Q值表

        参数:
            state (int): 当前状态
            action (int): 执行的动作
            reward (float): 奖励
            next_state (int): 下一个状态
            done (bool): 是否结束

        """
        q_value = self.q_table[state, action]
        if done:
            q_value += self.learning_rate * (reward - q_value)
        else:
            q_value += self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[next_state]) - q_value)
        self.q_table[state, action] = q_value


def train_rl_agent():
    """
    训练强化QLearning算法模型

    返回:
        agent (QLearningAgent): 训练好的模型
    """
    env = StockTradingEnvironment()
    num_states = env.num_days + 1
    num_actions = 2
    agent = QLearningAgent(num_states, num_actions)
    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
    return agent


if __name__ == "__main__":
    agent = train_rl_agent()
    env = StockTradingEnvironment()
    state = env.reset()
    done = False
    actions = []  # 记录每一天的行动
    while not done:
        action = agent.select_action(state)
        actions.append(action)
        next_state, _, done, _ = env.step(action)
        state = next_state
    env.render()

    # 打印最优的每日执行策略
    print("最优的每日执行策略:")
    for day, action in enumerate(actions):
        print("Day {}: {}".format(day+1, "买入" if action == 1 else "持有"))
