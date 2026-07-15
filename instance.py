import numpy as np
from itertools import product


class Instance:
    """论文中固定时域 (C)C-POMDP 模型的接口。

    对应论文第 2 节的 M=<S,A,O,T,O,U,b0,h>；子类负责给出状态、动作、
    观测以及转移/观测/效用/风险模型。求解器只依赖这个接口，因此更换问题实例
    时不需要修改 ILP 和 HILP 的主体。
    """

    name = ""
    type = "min"  # 目标类型：实验中的 Grid 是最小化代价，Tiger 是最大化效用
    states = []  # S
    observations = []  # O
    actions = []  # A
    horizon = 3  # h，即最多执行的动作步数

    # b0：初始信念。字典只保存非零概率状态，避免在大网格上存大量 0
    b0 = {}  # 初始信念 b0，仅保存概率非零的状态
    goal_state = 0  # 目标状态；Grid 到达该状态后单步代价为 0

    # 反向邻接表：states_reachable_to[s] 保存一步可能转移到 s 的前驱状态，
    # 用于加速 solver._update_safe_belief 中的预测分布计算。
    states_reachable_to = {}  # 反向邻接表：目标状态 -> 一步可到达它的前驱状态集合

    delta = 0  # 论文中的风险预算 Delta
    risk_states = states  # R：可能产生风险的状态集合。只遍历该集合可加速风险概率求和。


    def trans_model(self, s, a):
        return {}

    def obs_model(self, s, a):
        return {}

    def reward_model(self, s, a):
        return {}

    def risk_model(self, s, a):
        return {}

    def duration_model(self,q):
        """返回历史 q 已执行的动作数。

        当前仓库只实现论文实验中的固定单位时长模型 D(s,a)=1，所以历史
        (a1,o1,a2,...) 的时长就是动作个数。论文中的状态相关期望时长和
        高斯随机时长需要额外实现 tau(q)，此处尚未包含。
        """
        return np.floor((len(q)+1)/2)

    def reward_heuristic(self, s, a):
        # HILP 前沿节点的默认效用/代价启发式；子类可提供更强的可采纳估计。
        return self.reward_model(s,a)

    def risk_heuristic(self,s,a):
        pass

    # helpers
    def action_to_string(self, a):
        return ""

    def print_state(self):  # user friendly state message
        pass


class GridInstance(Instance):
    """论文第 5 节的部分可观测 Grid 实验（代码实现的是固定单位时长版）。"""

    name = "Grid"
    grid_size = (4, 4)
    occupancy = None  # 危险状态占用矩阵：1 表示危险格，0 表示安全格
    actions = [0, 1, 2, 3]  # 动作编号：左、上、右、下（顺时针）
    observations = [0, 1, 2]  # 可观测到的相邻边界墙数量
    risk_states = None  # 危险状态集合 R，实例化时由 risk_idx 给出
    u_heuristic_grid = None  # 每个网格状态到目标的启发式剩余代价表

    # TODO allow walls within grid

    def __init__(self, size=(5, 5), risk_idx=[(0,0), (3, 0), (3, 1),(1,3),(1,4)], start_s=(4, 0), goal_s=(0, 4), delta = .1, u_h = "manhattan"):
        # 论文采用从 1 开始的坐标；代码采用从 0 开始的 (行, 列) 坐标。
        # 默认参数正好对应论文的 5x5 网格、5 个危险状态、左下起点和右上目标。
        self.grid_size = size  # 当前网格尺寸 (行数, 列数)
        self.start_state = start_s  # 智能体起始状态坐标
        self.b0 = {start_s: 1}  # 初始位置完全确定
        self.goal_state = goal_s  # 智能体目标状态坐标
        self.occupancy = np.zeros(size)  # 危险格指示矩阵，初始全部为安全格
        for (i, j) in risk_idx:
            self.occupancy[i, j] = 1  # 将 risk_idx 中的坐标标记为危险格
        self.states = [(i, j) for i, j in product(range(size[0]), range(size[1]))]  # 全部网格坐标组成的状态集合 S
        self.risk_states = risk_idx  # 可能导致任务失败的危险状态集合 R
        self.delta = delta  # CC-POMDP 允许的最大累计执行风险 Delta


        # 论文 HILP 实验使用 Manhattan heuristic。这里预先计算每个状态到
        # 目标的距离，作为尚未展开的 frontier 节点的剩余代价估计 h_q。
        self.u_heuristic_grid = np.zeros(size)  # h^u 的查找表：每格到目标的估计剩余代价
        for i in np.arange(size[0]):
            for j in np.arange(size[1]):
                if u_h == "manhattan":
                    self.u_heuristic_grid[i][j] = np.abs(i - self.goal_state[0]) + np.abs(j - self.goal_state[1])  # Manhattan 剩余距离
                else:
                    self.u_heuristic_grid[i][j] =  np.sqrt((i - self.goal_state[0]) ** 2 + (j - self.goal_state[1]) ** 2)  # Euclidean 剩余距离

        # 为每个目标状态建立一步前驱集合，供贝叶斯预测时使用。
        for s in self.states:
            self._update_states_reachable_to(s)


    def trans_model(self, s, a):
        """论文实验的 T(s,a,s')：0.85 按目标方向，左右偏移各 0.075。"""

        i = s[0]  # 当前状态的行坐标
        j = s[1]  # 当前状态的列坐标
        if a == 0 or a == 2:  # horizontal
            if a == 0:  # left
                same = (i, j - 1 if j > 0 else 0)  # 按期望方向移动后的状态
            if a == 2:  # right
                same = (i, j + 1 if j < self.grid_size[1] - 1 else self.grid_size[1] - 1)  # 按期望方向移动后的状态
            left = (i + 1 if i < self.grid_size[0] - 1 else self.grid_size[0] - 1, j)  # 相对期望方向向左偏移后的状态
            right = (i - 1 if i > 0 else 0, j)  # 相对期望方向向右偏移后的状态
        if a == 1 or a == 3:  # vertical
            if a == 1:  # up
                same = (i - 1 if i > 0 else 0, j)  # 按期望方向移动后的状态
            if a == 3:  # down
                same = (i + 1 if i < self.grid_size[0] - 1 else self.grid_size[0] - 1, j)  # 按期望方向移动后的状态
            left = (i, j - 1 if j > 0 else 0)  # 相对期望方向向左偏移后的状态
            right = (i, j + 1 if j < self.grid_size[1] - 1 else self.grid_size[1] - 1)  # 相对期望方向向右偏移后的状态

        # 碰到边界时机器人留在原格；若目标方向和某个偏移方向重合，
        # 两条概率质量必须合并，保证返回分布之和仍为 1。
        if same == right:
            next = {same: .85 + 0.075, left: .075}  # 下一状态概率分布 T(s,a,.)，合并重合分支
        elif same == left:
            next = {same: .85 + 0.075, right: .075}  # 下一状态概率分布 T(s,a,.)，合并重合分支
        else:
            next = {same: .85, left: .075, right: .075}  # 下一状态概率分布 T(s,a,.)

        return next

    def obs_model(self, s, a):
        """论文中的观测模型：只能看到相邻边界墙数量 0/1/2。

        真实墙数以 0.85 被正确观测，另外两个观测各以 0.075 出现。
        观测与动作无关，但保留 a 参数以符合统一的 O(o,s,a) 接口。
        """

        (i, j) = s  # 当前状态的行、列坐标
        m,n = self.grid_size[0] -1, self.grid_size[1]-1  # 最大行、列下标，用于判断边界和角落
        if s == (m, n) or s == (0, 0) or s == (m,0) or s == (0,n):
            return {2: .85, 1: 0.075, 0: 0.075}
        elif i == 0 or i == m or j == 0 or j == n:
            return {2: 0.075, 1: .85, 0: 0.075}
        else:
            return {2: 0.075, 1: 0.075, 0: .85}

    def reward_model(self, s, a):
        # 每个非目标状态执行一步的代价为 1；到达目标后代价为 0。
        # 因而最小化累计期望代价等价于尽量少走步骤到达目标。
        if s == self.goal_state: # or s in self.risk_states:
            return 0
        return 1
        # (i,j) = s
        #
        # if a == 0: #left
        #     next = (i,j-1 if j > 0 else 0)
        # elif a == 1: # up
        #     next = (i-1 if i > 0 else 0, j)
        # elif a == 2: # right
        #     next = (i, j + 1 if j < self.grid_size[1] - 1 else self.grid_size[1] - 1)
        # elif a == 3:  # down
        #     next = (i + 1 if i < self.grid_size[0] - 1 else self.grid_size[0] - 1, j)
        # if next == self.goal_state:
        #     return 0
        # return 1

    def reward_heuristic(self, s, a):
        # 忽略转移噪声和部分可观测性后的最短网格距离，用来引导 HILP 扩展。
        # return np.sqrt((s[0] - self.goal_state[0])**2 + (s[1] - self.goal_state[1])**2)
        return self.u_heuristic_grid[s[0]][s[1]]

        # (i,j) = s

        # if a == 0: #left
        #     next = (i,j-1 if j > 0 else 0)
        # elif a == 1: # up
        #     next = (i-1 if i > 0 else 0, j)
        # elif a == 2: # right
        #     next = (i, j + 1 if j < self.grid_size[1] - 1 else self.grid_size[1] - 1)
        # elif a == 3:  # down
        #     next = (i + 1 if i < self.grid_size[0] - 1 else self.grid_size[0] - 1, j)
        #
        # return np.abs(next[0] - self.goal_state[0]) + np.abs(next[1] - self.goal_state[1])

    def risk_model(self, s, a=None):
        # R(s,a)：进入危险格时为 1，否则为 0；本实例中风险与动作无关。
        return self.occupancy[s[0]][s[1]]


    def _update_states_reachable_to(self, s):
        """缓存所有一步可能到达 s 的状态，用于稀疏信念更新。"""

        (i, j) = s  # 正在建立反向邻接表的目标状态坐标

        self.states_reachable_to[s] = [(i - 1, j) if i > 0 else (i, j),  # 一步可能到达 s 的候选前驱状态
                                       (i + 1, j) if i < self.grid_size[0] - 1 else (i, j),
                                       (i, j - 1) if j > 0 else (i, j),
                                       (i, j + 1) if j < self.grid_size[1] - 1 else (i, j)]
        if i == 0 or i == self.grid_size[0] - 1 or j == 0 or j == self.grid_size[1]:
            self.states_reachable_to[s].append(s)

        self.states_reachable_to[s] = set(self.states_reachable_to[s])  # 去除边界造成的重复前驱状态


    def action_to_string(self, a):
        if a == 0:
            return "←"
        if a == 1:
            return "↑"
        if a == 2:
            return "→"
        if a == 3:
            return "↓"

    def print_state(self):
        print("┌", end='')
        for j in np.arange(self.grid_size[1]):
            print("──", end='')
        print("─┐")
        for i in np.arange(self.grid_size[0]):
            print("│ ", end='')
            for j in np.arange(self.grid_size[1]):
                if (i, j) == self.start_state:
                    print("S ", end='')
                elif (i, j) == self.goal_state:
                    print("G ", end='')
                elif self.occupancy[i, j] == 0:
                    print(". ", end='')
                elif self.occupancy[i, j] == 1:
                    print("🔥", end='')  # prints fire, but not visible here
            print("│")  # new line
        print("└", end='')
        for j in np.arange(self.grid_size[1]):
            print("──", end='')
        print("─┘")



class TigerInstance(Instance):
    """经典 Tiger POMDP 的简化调试实例，不是论文 Grid 实验的主体。"""

    type = 'max'  # Tiger 问题最大化累计效用
    name = "Tiger"  # 调试实例名称
    states = ["tiger-left", "tiger-right"]  # 老虎位于左门或右门的状态集合
    actions = ["open-left", "open-right", 'listen']  # 开左门、开右门、监听
    observations = ["tiger-left", "tiger-right"]  # 监听后关于老虎位置的观测

    risk_states = states  # 两种状态都可能因开错门产生风险

    # trans_prob = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1.0, 0.0], [0.0, 1.0]]])  # np.array
    # obs_prob = np.array(
    #     [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [[0.85, 0.15], [0.15, 0.85]]])  # np.array

    states_reachable_to = {0:[0,1], 1:[0,1]}  # 每个状态均可能一步转移到另一状态
    b0 = {0:.5, 1:.5}  # 初始时老虎位于左右两侧的概率各为 0.5

    def trans_model(self, s, a):
        # return {0: self.trans_prob[s][a][0], 1: self.trans_prob[s][a][1]}
        return {0:.5, 1:.5} if a != 2 else {s:1}

    def obs_model(self, s, a):
        return {s: 85, 1-s: 15} if a == 2 else {0:.5, 1:.5}


    def reward_model(self, s, a):
        if a == 2:
            return -1
        elif a == s:
            return -100
        else:
            return 10

    def risk_model(self, s, a):
        return s == a
    # def reward_heuristic(self, s, a):
    #     return self.reward_model(s,a)
