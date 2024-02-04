# Stock-Purchase-Optimization-Project
# 项目介绍
客户需要银行在6个月内购买100万股。每个月有22个交易日，银行可以在每个月的最后一个交易日（第22、44、66、88、110、132天）终止执行，如果已购买100万股。假设资产在第一天和银行终止执行的当天之间的平均价格为p。那么银行的利益将为（p - d）* 100万，减去购买100万股的成本。资产价格S(t)遵循几何布朗运动，其中dS(t)/S(t) = σdw(t)，S(0) = 100，σ = 40%，Wt是标准布朗运动。
(1) 找到最优的每日执行策略以最大化银行的利益。（强化学习）
(2) 找到银行的预期利益和使预期值为0的折现率d。（蒙特卡洛模拟）
# 关键词
1. 股票购买
2. 强化学习
3. 几何布朗运动
4. 折现率
5. 蒙特卡洛模拟
6. 交易日
7. 资产价格路径
8. Q-learning
# 项目思路
## 强化学习思路
在这个项目中，我们采用强化学习作为优化银行购买策略的方法。以下是项目的强化学习思路：
1. 定义状态空间：
  - 将每个交易日看作一个状态，即总共有132个状态（6个月，每个月22个交易日）。
  - 状态表示：可以包括当前股票价格、已购买股数等信息。
2. 定义动作空间：
  - 每个状态下银行可以采取的动作包括购买股票的数量或者不购买。
  - 动作选择的离散化：确定每次购买的股票数量的离散选择。
3. 定义奖励函数：
  - 奖励函数用于评估银行每个决策的好坏。
  - 奖励可以基于利润，定义为购买后的股票卖出利润减去购买成本。
4. 定义状态转移概率：
  - 根据股票价格的变动模型（几何布朗运动），计算每个状态之间的转移概率。
  - 转移概率可以基于当前股价和前一天的股价，使用几何布朗运动的性质计算。
5. 使用强化学习算法进行训练：
  - 使用Q-learning、SARSA等强化学习算法。
  - 在每个交易日，银行根据当前状态选择最优动作，执行并更新Q值。
  - 通过与环境交互，不断迭代学习，更新策略和价值函数，以找到最优的每日执行策略。
## 蒙特卡洛模拟思路
蒙特卡洛模拟是为了找到银行的预期利益和确定使预期值为0的折现率。以下是项目的蒙特卡洛模拟思路：
1. 定义模拟时间段：
  - 设定模拟的时间段，即6个月的交易日。
  - 每个月有22个交易日，总共模拟132个交易日。
2. 生成资产价格路径：
  - 使用几何布朗运动模型生成资产价格的路径。
  - 利用随机数生成器模拟价格的变动，确保符合给定的几何布朗运动方程。
3. 计算银行的利益：
  - 对于每个交易日，根据银行的执行策略计算银行的利益。
  - 利益计算基于购买后的股票卖出利润减去购买成本。
4. 多次模拟并累加利益：
  - 重复进行多次模拟，例如1000次，以获得对于每次模拟的不同资产价格路径下的银行利益。
  - 累加每次模拟的银行利益，以及计算平均值作为预期利益的估计值。
5. 寻找使预期值为0的折现率d：
  - 在模拟中尝试不同的折现率d。
  - 对于每个折现率d，计算模拟中的预期利益。
  - 通过二分法或其他优化算法，找到使预期值为0的折现率d的估计值。
# 具体代码
StockTradingEnvironment是一个OpenAI Gym的环境类，用于模拟股票交易的环境。它包括了一个股票价格的模型，允许代理根据环境状态执行买入或持有的动作，并根据这些动作获得奖励。该环境类可用于强化学习算法的训练，例如Q-learning或深度强化学习，以优化股票交易策略。

```python
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

```

QLearningAgent 类实现了一个简单的Q-learning算法，用于在给定的状态空间和动作空间中学习最优策略。它通过使用Q值表来表示状态和动作的关联，并通过不断地与环境交互来更新Q值，从而优化策略。其中，探索率参数决定了在选择动作时是进行探索还是利用已学到的信息。

```python
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

```

train_rl_agent 函数：
- 用途：训练强化学习模型，具体是一个基于 Q-learning 的智能体。
- 实现：
  - 创建一个股票交易环境 StockTradingEnvironment。
  - 初始化 Q-learning 智能体 QLearningAgent，指定状态和动作的数量。
  - 循环进行训练：
    - 重置环境状态，开始新的一轮训练。
    - 在每个时间步内，根据当前状态选择动作，与环境交互，更新 Q 值。
    - 循环直至交易结束。
  - 返回训练好的智能体。
主程序部分 (if name == "__main__")：
- 创建一个股票交易环境 StockTradingEnvironment。
- 重置环境状态。
- 循环执行训练好的 Q-learning 智能体的策略，记录每一天的动作。
- 打印当前环境状态（交易天数、股票价格、持有的股票数量）。
- 打印最优的每日执行策略，根据训练好的 Q-learning 模型选择动作。

```python
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

```

这是一个名为 AssetPriceGenerator 的类，用于生成模拟资产价格的路径，其主要目的是通过模拟几何布朗运动来生成具有漂移和波动性的资产价格。以下是对该类的各部分的详细解释：
init 方法：
- 作用：初始化资产价格生成器。
- 参数：
  - num_days（int）：模拟的天数，即资产价格路径的长度。
  - initial_price（float）：初始资产价格。
  - drift（float）：漂移率，表示资产价格路径的趋势方向。
  - volatility（float）：波动率，表示资产价格路径的波动性。
generate_asset_price 方法：
- 作用：生成资产价格路径。
- 返回值：包含每天价格的列表。
- 实现：
  - 初始化 prices 列表，将初始价格添加到列表中。
  - 循环生成每一天的资产价格：
    - 获取前一天的价格。
    - 根据几何布朗运动模型生成资产价格的变动，考虑漂移和波动性。
    - 将生成的价格添加到列表中。
  - 返回包含所有天数价格的列表。
总体作用：
该类提供了一个模拟资产价格路径的工具，通过几何布朗运动模型生成考虑漂移和波动性的资产价格。

```python
class AssetPriceGenerator:
    def __init__(self, num_days, initial_price, drift, volatility):
        """
        初始化资产价格生成器

        参数:
        - num_days (int): 模拟的天数
        - initial_price (float): 初始价格
        - drift (float): 漂移率
        - volatility (float): 波动率
        """
        self.num_days = num_days
        self.initial_price = initial_price
        self.drift = drift
        self.volatility = volatility

    def generate_asset_price(self):
        """
        生成资产价格路径

        返回值:
        - prices (list): 包含每天价格的列表
        """
        prices = [self.initial_price]
        for _ in range(self.num_days - 1):
            price = prices[-1]
            # 根据几何布朗运动模型生成资产价格的变动
            price_change = self.drift * price + self.volatility * random.gauss(0, 1)
            price += price_change
            prices.append(price)
        return prices

```

BankProfitCalculator 类的总体作用是根据给定的资产价格路径、购买数量和交易成本，计算银行的利润。在金融领域中，银行作为交易者可能通过购买和持有资产来实现盈利，但同时需要考虑成本因素，比如交易成本。
具体而言，这个类提供了一个计算银行利润的方法，该方法通过模拟银行在资产市场上的买卖行为，考虑购买数量和交易成本，计算银行在给定资产价格路径下的利润情况。
主要函数解释：
1. init 函数：
  - 初始化银行利润计算器，接受两个参数：购买数量 (buy_quantity) 和交易成本 (cost)。
2. calculate_bank_profit 函数：
  - 计算银行的利润，接受一个参数 asset_prices，它是一个包含资产价格路径的列表。
  - 通过迭代资产价格路径，模拟银行的交易行为，根据购买数量和交易成本计算银行的利润。
  - 返回银行的利润 (bank_profit)，表示银行在给定资产价格路径下的盈亏情况。

```python
class BankProfitCalculator:
    def __init__(self, buy_quantity, cost):
        """
        初始化银行利润计算器

        参数:
        - buy_quantity (int): 购买数量
        - cost (float): 交易成本
        """
        self.buy_quantity = buy_quantity
        self.cost = cost

    def calculate_bank_profit(self, asset_prices):
        """
        计算银行的利润

        参数:
        - asset_prices (list): 资产价格路径

        返回值:
        - bank_profit (float): 银行的利润
        """
        num_shares = 0
        bank_profit = 0
        for i in range(len(asset_prices) - 1):
            price_today = asset_prices[i]
            price_tomorrow = asset_prices[i + 1]
            if self.buy_quantity > 0:
                # 购买股票
                num_shares += self.buy_quantity
                bank_profit -= self.buy_quantity * price_today
                bank_profit -= self.cost
            bank_profit += num_shares * price_tomorrow
        return bank_profit
```

MonteCarloSimulation 类的总体作用是进行蒙特卡洛模拟，通过多次模拟计算预期利润。在金融建模中，蒙特卡洛模拟常用于评估金融产品或策略在不同市场情景下的表现，通过随机生成多个可能的市场走势，从中获取预期的统计结果。
主要函数解释：
1. init 函数：
  - 初始化蒙特卡洛模拟器，接受一个参数 num_simulations，表示进行模拟的次数。
2. simulate 函数：
  - 进行蒙特卡洛模拟，计算预期利润。
  - 接受多个参数，包括模拟的天数 (num_days)、初始价格 (initial_price)、漂移率 (drift)、波动率 (volatility)、购买数量 (buy_quantity)、交易成本 (cost)。
  - 在每次模拟中，使用 AssetPriceGenerator 生成资产价格路径，并使用 BankProfitCalculator 计算银行的利润。
  - 模拟多次后，计算总利润的平均值，作为预期利润。
  - 返回预期利润 (expected_profit)。

```python
class MonteCarloSimulation:
    def __init__(self, num_simulations):
        """
        初始化蒙特卡洛模拟器

        参数:
        - num_simulations (int): 模拟次数
        """
        self.num_simulations = num_simulations

    def simulate(self, num_days, initial_price, drift, volatility, buy_quantity, cost):
        """
        进行蒙特卡洛模拟，计算预期利润

        参数:
        - num_days (int): 模拟的天数
        - initial_price (float): 初始价格
        - drift (float): 漂移率
        - volatility (float): 波动率
        - buy_quantity (int): 购买数量
        - cost (float): 交易成本

        返回值:
        - expected_profit (float): 预期利润
        """
        total_profit = 0
        asset_price_generator = AssetPriceGenerator(num_days, initial_price, drift, volatility)
        bank_profit_calculator = BankProfitCalculator(buy_quantity, cost)

        for _ in range(self.num_simulations):
            asset_price_path = asset_price_generator.generate_asset_price()

            # 根据执行策略计算银行的购买数量
            # 这里可以根据具体策略进行调整
            # 例如，可以使用强化学习算法训练一个模型来决定购买数量
            # 这里仅作为示例，随机选择购买数量
            buy_quantity = random.randint(0, 10)

            # 计算银行的利润
            profit = bank_profit_calculator.calculate_bank_profit(asset_price_path)

            total_profit += profit

        expected_profit = total_profit / self.num_simulations
        return expected_profit
```

DiscountRateEstimator 类的总体作用是通过二分法估计使预期值为零的折现率。在金融建模中，折现率是一个重要的参数，用于将未来现金流的价值折算为当前值，以进行综合的财务分析。
主要函数解释：
1. init 函数：
  - 初始化折现率估计器，接受两个参数：num_simulations 表示模拟次数，epsilon 表示估计的精度。
2. estimate_discount_rate 函数：
  - 估计使预期值为0的折现率。
  - 接受多个参数，包括模拟的天数 (num_days)、初始价格 (initial_price)、漂移率 (drift)、波动率 (volatility)、购买数量 (buy_quantity)、交易成本 (cost)。
  - 使用二分法在给定的折现率范围内搜索，找到使预期值为0的折现率。
  - 返回估计的折现率 (discount_rate)。

```python
class DiscountRateEstimator:
    def __init__(self, num_simulations, epsilon=0.0001):
        """
        初始化折现率估计器

        参数:
        - num_simulations (int): 模拟次数
        - epsilon (float): 估计精度
        """
        self.num_simulations = num_simulations
        self.epsilon = epsilon

    def estimate_discount_rate(self, num_days, initial_price, drift, volatility, buy_quantity, cost):
        """
        估计使预期值为0的折现率

        参数:
        - num_days (int): 模拟的天数
        - initial_price (float): 初始价格
        - drift (float): 漂移率
        - volatility (float): 波动率
        - buy_quantity (int): 购买数量
        - cost (float): 交易成本

        返回值:
        - discount_rate (float): 估计的折现率
        """
        lower_bound = 0
        upper_bound = 1

        while upper_bound - lower_bound > self.epsilon:
            discount_rate = (lower_bound + upper_bound) / 2
            monte_carlo_simulation = MonteCarloSimulation(self.num_simulations)
            expected_profit = monte_carlo_simulation.simulate(num_days, initial_price, drift, volatility, buy_quantity, cost)
            discounted_profit = expected_profit / (1 + discount_rate) ** num_days

            if discounted_profit > 0:
                lower_bound = discount_rate
            else:
                upper_bound = discount_rate

        return discount_rate
```

该主程序的目的是使用蒙特卡洛模拟计算预期利润，并通过折现率估计器找到使预期值为零的折现率。
1. 设置参数：
  - 定义了蒙特卡洛模拟的参数，包括模拟次数 (num_simulations)、模拟的天数 (num_days)、初始价格 (initial_price)、漂移率 (drift)、波动率 (volatility)、购买数量 (buy_quantity)、交易成本 (cost)。
2. 创建折现率估计器：
  - 使用 DiscountRateEstimator 类创建了一个折现率估计器 (discount_rate_estimator)，并传入模拟次数作为参数。
3. 蒙特卡洛模拟计算预期利润：
  - 使用 MonteCarloSimulation 类创建了一个蒙特卡洛模拟器 (monte_carlo_simulation)，并传入模拟次数作为参数。
  - 调用 simulate 方法进行模拟，计算预期利润 (expected_profit)。
4. 寻找使预期值为0的折现率：
  - 调用折现率估计器的 estimate_discount_rate 方法，传入各个参数，找到使预期值为零的折现率 (discount_rate)。
5. 输出结果：
  - 打印预期利润和折现率。

```python
if __name__ == "__main__":
    # 设置参数
    num_simulations = 1000
    num_days = 180
    initial_price = 100
    drift = 0.05
    volatility = 0.2
    buy_quantity = 5
    cost = 10

    # 创建折现率估计器
    discount_rate_estimator = DiscountRateEstimator(num_simulations)

    # 进行蒙特卡洛模拟，计算预期利润
    monte_carlo_simulation = MonteCarloSimulation(num_simulations)
    expected_profit = monte_carlo_simulation.simulate(num_days, initial_price, drift, volatility, buy_quantity, cost)

    # 寻找使预期值为0的折现率
    discount_rate = discount_rate_estimator.estimate_discount_rate(num_days, initial_price, drift, volatility, buy_quantity, cost)

    # 输出结果
    print("预期利润:", expected_profit)
    print("折现率d:", discount_rate)
```
项目链接
CSDN：https://blog.csdn.net/m0_46573428/article/details/136034838

后记
如果觉得有帮助的话，求 关注、收藏、点赞、星星 哦！
