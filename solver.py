import numpy as np
import networkx as nx
import instance as ins
import itertools
import gurobipy as gb
import time



def FPTAS(ins: ins.Instance, T: nx.DiGraph, epsilon = .5, bound = None):
    """早期的近似动态规划实验代码，并非论文 Algorithm 1--3 的主流程。

    该函数依赖动作/观测必须是连续整数等额外假设，且当前 Notebook 没有调用它。
    阅读论文实现时，应重点看 preprocess、ILP、p_ILP 和 heuristic_search。
    """

    # 将效用按 K=epsilon*P/n 缩放取整，把连续效用预算变成有限 DP 维度。
    n = len([k[1] for k in T.nodes("u") if k[1] is not None])  # 具有局部效用 u_q 的动作节点数量
    P = max([k[1] for k in T.nodes("u") if k[1] is not None])  # 所有动作节点中的最大局部效用
    print(n*P)
    K = epsilon*P/n  # FPTAS 的效用离散化步长
    print("K ", K)
    # rounding
    for q in T.nodes():
        if T.nodes[q]['t'] == 'a':
            # print(q)
            T.nodes[q]["u_"] = np.ceil(T.nodes[q]['u']/K)  # 缩放取整后的节点效用
            # print("u = {} --> u_ = {}".format(T.nodes[q]["u"], T.nodes[q]["u_"]))
    print("max = " , max([k[1] for k in T.nodes("u_") if k[1] is not None]))
    if bound is None:
        # VV = np.ceil(n*P/K)
        VV = np.floor(n*P/K)  # 未指定上界时的最大离散效用预算
    else:
        # VV = np.ceil(bound/K)
        VV = np.floor(bound / K)  # 指定效用上界对应的最大离散预算
    print(VV)
    L = np.arange(VV+1)  # 动态规划需要枚举的离散效用预算集合
    # L = L[int(VV+1):0:-1] # reverse L for min objective

    D = {}  # DP 表：(历史 q, 遍历阶段 i, 效用预算 l) -> 最小累计风险
    q_count = {}  # 每个 (历史, 效用预算) 在带回访 DFS 中被访问的次数
    for l in L:
        print("l = ", l)
        D[(q,0,l)] = np.infty  # 当前效用预算下的 DP 单元默认标记为不可行


        for q in AND(ins,()):
            # print("q: {} | l: {}".format(q, l))
            # print(D)
            i = q_count.get((q,l),0)  # 当前历史在该预算下的 DFS 回访阶段
            if T.nodes[q]['t'] == 'o' and i == 0:
                D[(q,i,l)] = np.infty  # 观测节点首次访问时尚未汇总任何动作分支
            elif len(q) == 1 and i == 0 and T.nodes[q]['t'] == 'a':
                if T.nodes[q]['u_'] <= l:
                    D[(q,0,l)] = T.nodes[q]['r']  # 根动作满足效用预算时，以其局部风险初始化
                else:
                    D[(q,0,l)] = np.infty  # 根动作超出效用预算，因此该 DP 状态不可行
            elif len(q) > 1 and i == 0 and T.nodes[q]['t'] == 'a':
                # TODO: Huge assumption! Observations are ordered numbers 0,1,2
                obs = q[-2:-1][0]  # 当前动作历史的上一个观测编号
                if D[(q[:-2],max(obs - 1, 0), max(l - T.nodes[q]['u_'],0) ) ] <= ins.delta:
                    D[(q,0,l)] = D[(q[:-2], max(obs - 1, 0), max(l - T.nodes[q]['u_'], 0))]  # 从父历史和剩余预算继承风险值
                else:
                    D[(q,0,l)] = np.infty  # 父历史风险超出 Delta，当前状态不可行
            elif i>0 and T.nodes[q]['t'] == 'o': # R4
                # TODO: Another assumption
                act = i-1  # 当前回访阶段对应的候选动作编号
                if ins.duration_model(q+(act,)) < ins.horizon:
                    D[(q,i,l)] =  min(D[(q,i-1,l)], D[(q+(act,), act, l)] )  # 在已有动作与第 i 个候选动作间取较小风险
                else:
                    D[(q,i,l)] =  min(D[(q,i-1,l)], D[(q+(act,), 0, l)] )  # 终端动作不再递归，直接比较其风险

            elif i>0 and T.nodes[q]['t'] == 'a' and ins.duration_model(q) < ins.horizon: #R5
                # TODO: Another assumption
                act = i-1  # 当前回访阶段对应的候选动作编号
                D[(q,i,l)] = D[(q+(act,), act, l)]  # 动作节点继续沿对应观测/动作分支传递 DP 值

            q_count[(q,l)] = i + 1  # 记录下一次回访该历史时所处的遍历阶段
    return D

# FPTAS 使用的带回访深度优先遍历：AND 节点枚举动作，OR 节点枚举观测。
def AND(ins: ins.Instance, p):
    for a in np.arange(len(ins.actions)):
        q = p + (a,)  # 在父历史 p 后追加动作 a 得到的动作历史
        yield q
        if ins.duration_model(q) <  ins.horizon:
            for c in OR(ins, q):
                yield c
        yield p
def OR(ins: ins.Instance, p):
    for o in np.arange(len(ins.observations)):
        q = p + (o,)  # 在动作历史 p 后追加观测 o 得到的观测历史
        yield q
        for c in AND(ins, q):
            yield c
        yield p

# def AND_OR_Traverse(ins: ins.Instance, p):
#     yield p
#     for a in ins.actions:
#         q = p + (ins.action_to_string(a),)
        # q = p + (a,)
        # yield q
        # if ins.duration_model(q) < ins.horizon:
        #     for o in ins.observations:
        #         c = q + (o,)
        #         for _c in AND_OR_Traverse(ins, c):
        #             yield _c
        #         yield q
        #         yield c
        # else:
        #     yield p
    # yield p

def preprocess(ins: ins.Instance):
    """展开完整历史树，并计算 ILP 所需的节点常数。

    这对应论文 Algorithm 1 ``Preprocess`` 和 Algorithm 2 ``Expand`` 在
    CC-POMDP、固定单位时长情形下的实现。历史 q 用扁平元组表示，例如
    (a1, o1, a2)：奇数长度是动作节点 q in A~，偶数长度是观测节点 q in O~。

    节点属性：
      b   安全信念，即已知此前未进入风险状态时的条件状态分布；
      rho 安全历史出现概率 rho~(q)；
      u   论文 Lemma 3.3 的局部期望效用/代价 u_q；
      r   Lemma 3.3 的局部执行风险 r_q；
      h   HILP 在搜索边界使用的剩余代价启发式 h^u_q。

    返回完整 AND-OR 图 T，以及每个动作历史对应的 ILP 变量索引。
    """

    T = nx.DiGraph()  # 完整 AND-OR 历史树，节点保存 b/rho/u/r/h 等属性
    # 空历史 q=0 是观测层根节点。初始时历史必然发生，且假设 b0 安全。
    T.add_node((), b=ins.b0, rho=1, t='o')
    size_of_tree_ending_with_obs = sum([len(ins.actions)**i*len(ins.observations)**(i) for i in range(1,ins.horizon+1)]) + 1  # 理论上的观测历史数量（含根）
    print("size of tree ending up with obs = ", size_of_tree_ending_with_obs)

    size_of_tree_ending_with_action = sum(  # 理论上的动作历史数量（含额外计数 1）
        [len(ins.actions) ** i * len(ins.observations) ** (i - 1) for i in range(1, ins.horizon + 1)]) + 1
    print("number of action vars = ", size_of_tree_ending_with_action - 1)
    size_of_and_or_tree = sum(  # 实际生成的动作节点与非终端观测节点总数
        [len(ins.actions) ** i * len(ins.observations) ** (i - 1) for i in range(1, ins.horizon + 1)]) \
                          + sum([len(ins.actions) ** i * len(ins.observations) ** i for i in range(1, ins.horizon)]) + 1
    print("size of AND-OR tree = ", size_of_and_or_tree)

    ILP_vars = []  # 所有动作历史 q；每个 q 对应一个策略决策变量 x_q
    # AND_OR 按 a,o,a,o... 递归产生所有未达到 horizon 的历史。
    for q in AND_OR(ins, ()):
        if len(q) != 0:
            T.add_edge(q[:-1], q)
            parent = [i for i in T.predecessors(q)][0]  # 当前历史 q 的直接父历史
            # print(parent, "-->", q)
            # print('parent belief = ', T.nodes[parent]['b'])

            if len(q) % 2 == 0:  # 偶数长度：(a1,o1,...,ak,ok)，观测节点
                T.nodes[q]['t'] = 'o'  # 节点类型：observation/观测节点
                grand_parent = [i for i in T.predecessors(parent)][0]  # 跨过父动作节点后的上一观测历史
                g_rho = T.nodes[grand_parent]['rho']  # 上一观测历史安全发生的概率 rho~
                # 在动作节点的预测信念上结合当前观测，得到 Eq. (11) 的
                # 安全后验；p 是在安全条件下看到该观测的概率。
                b, p = _update_safe_belief(ins, safe_belief=T.nodes[parent]['b'], o=q[-1:][0])  # b：安全后验；p：条件观测概率
                p_action = q[-2:-1][0]  # 产生当前观测的父动作编号

                # 该实现用当前安全信念估计“本步仍安全”的概率，并递推
                # rho~(q)。源码已标注此处与论文公式仍需进一步核对。
                safe_at = 1 - sum([b.get(s, 0) * ins.risk_model(s, p_action) for s in ins.risk_states])  # 当前分支本步保持安全的条件概率
                ## TODO require equation modification !!
                # g_b = T.nodes[grand_parent]['b']
                # safe_at
                #
                # = 1 - sum([g_b.get(s, 0) * ins.risk_model(s, p_action) for s in ins.risk_states])
                T.nodes[q]['p'] = p  # 当前观测在安全条件下出现的概率
                T.nodes[q]['rho'] = g_rho * p * safe_at  # 当前安全历史的发生概率 rho~(q)
                # T.nodes[q]['safe_at'] = safe_at

            else:  # 奇数长度：(a1,o1,...,ak)，动作节点，也是 ILP 变量
                T.nodes[q]['t'] = 'a'  # 节点类型：action/动作节点，对应 ILP 变量
                p_rho = T.nodes[parent]['rho']  # 父观测历史安全发生的概率 rho~
                p_b = T.nodes[parent]['b']  # 执行动作前的安全后验信念
                action = q[-1:][0]  # 当前动作历史最后一个动作 a_q
                # 先执行动作得到安全预测信念。局部风险 r_q 是“安全到达该
                # 历史”的概率乘以本步落入风险集合的条件概率。
                b = _update_safe_belief(ins, safe_belief=T.nodes[parent]['b'], a=action)  # 动作执行后的安全预测信念
                T.nodes[q]['r'] = p_rho * sum([b.get(s, 0) * ins.risk_model(s, action) for s in ins.risk_states])  # 局部执行风险 r_q
                # u_q = rho~(q) * E[U(S_q,a_q)]，对应论文 Lemma 3.3。
                T.nodes[q]['u'] = p_rho * sum([v * ins.reward_model(s, action) for s, v in p_b.items()])  # 局部期望效用/代价 u_q
                # T.nodes[q]['h'] = p_rho * sum([v*ins.reward_heuristic(s,action) for s, v in b.items()])
                # 仅在搜索边界使用的 Manhattan 剩余代价估计；(1-Delta)
                # 是该仓库为风险预算加入的经验缩放，并非通用论文公式。
                T.nodes[q]['h'] = p_rho * (1-ins.delta) * sum([v*ins.reward_heuristic(s,action) for s, v in p_b.items()])  # 前沿剩余代价启发式 h^u_q
                ILP_vars.append(q)
            T.nodes[q]['b'] = b  # 节点 q 对应的安全信念分布
            # print('node belief = ', T.nodes[q]['b'])

    print("# nodes = ", len(T))
    return T, ILP_vars


def AND_OR(ins: ins.Instance, p):
    """生成完整 AND-OR 历史树，对应论文 Fig. 1。

    观测历史 p 是 OR 节点：策略选择一个动作；动作历史 q 是 AND 节点：
    环境可能产生任一观测，因此每个观测分支都必须出现在条件计划中。
    """

    for a in np.arange(len(ins.actions)):
        # q = p + (ins.action_to_string(a),)
        q = p + (a,)  # 从观测历史 p 选择动作 a 后形成的动作历史
        yield q
        if ins.duration_model(q) < ins.horizon:
            for o in np.arange(len(ins.observations)):
                c = q + (o,)  # 动作历史 q 产生观测 o 后形成的子历史
                yield c
                for _c in AND_OR(ins, c):
                    yield _c

def ILP(ins: ins.Instance, T: nx.DiGraph, var_idx: list, continuous=False ):
    """在完整历史树上求解论文式 (7)--(8) 的 ILP。

    每个动作历史 q 对应变量 x_q：x_q=1 表示该动作节点被选入确定性策略树。
    目标函数累加 u_q，风险约束累加 r_q；树结构约束保证根只选一个动作，
    且任何已选动作后的每种观测都恰好选择一个后继动作。

    注意：原仓库的 continuous=True 分支仍创建 BINARY 变量，因此这里目前
    实际只求确定性策略；论文 Remark 1 所述随机策略 LP 松弛尚未在此生效。
    """

    t1 = time.time()  # 完整 ILP 求解开始时间
    m = gb.Model("ILP")  # Gurobi 完整整数规划模型
    m.setParam("OutputFlag", 0)
    x={}  # 动作历史 q -> Gurobi 策略变量 x_q
    for q in var_idx:
        # x_q 是论文中的策略流变量。动作历史而非状态作为索引，使策略能够
        # 根据完整观测历史作决定。
        x[q] = m.addVar(vtype=gb.GRB.BINARY, name=str(q)) if not continuous else   m.addVar(vtype=gb.GRB.BINARY, name=str(q))  # x_q=1 表示动作历史 q 被策略选中
    # obj = gb.quicksum([x[q]*T.nodes[q]['u'] for q in var_idx])
    obj = gb.quicksum(  # 完整策略树的累计期望目标：内部节点用 u_q，终端节点用 h_q
        [x[q] * T.nodes[q]['u'] if ins.duration_model(q) < ins.horizon else x[q] * T.nodes[q]['h'] for q in var_idx])

    m.setObjective(obj, gb.GRB.MINIMIZE) if ins.type == "min" else m.setObjective(obj, gb.GRB.MAXIMIZE)




    # Definition 3.1 第一条：根观测节点只能选择一个起始动作。
    tree_c1 = gb.quicksum([x[(a,)] for a in np.arange(len(ins.actions))])  # 根节点所有候选起始动作变量之和
    m.addConstr(tree_c1 == 1, "tree_c1")

    for q in var_idx:
        if ins.duration_model(q) < ins.horizon: # replace with duration model
            for o in np.arange(len(ins.observations)):
                # Definition 3.1 第二条：若 q 被选中，则每个可能观测 o 后
                # 必须且只能选一个动作；若 q 未选中，其所有后继也为 0。
                tree_c2 = gb.quicksum([x[q+(o,a)] for a in np.arange(len(ins.actions))])  # 历史 q 看到 o 后所有后继动作变量之和
                m.addConstr(tree_c2 == x[q], "tree_c{}".format(q))
                m.update()

    # 式 (7) / Lemma 3.3：策略的总执行风险是各选中节点 r_q 的线性和。
    capacity_c = gb.quicksum([x[q]*T.nodes[q]['r'] for q in var_idx])  # 被选策略节点的累计执行风险
    m.addConstr(capacity_c <= ins.delta)

    m.update()
    m.optimize()

    return obj.getValue(),{k:v.x for k,v in x.items() if v.x > 0}, time.time()-t1


def p_ILP(m,ins: ins.Instance, T: nx.DiGraph, expanded: list, frontier = [], continuous=True, risk = False, warm_start = {}):
    """求解 HILP 每轮使用的部分整数规划 p-ILP（论文式 (13)）。

    expanded=E 保存已精确展开的动作节点，使用真实 u_q/r_q；frontier=F
    保存边界动作节点，目标中用启发式 h_q 代表尚未展开子树的剩余代价。
    warm_start 实际保存并复用已有 Gurobi 变量，避免每轮重建变量。
    """

    t1 = time.time()  # 本轮部分 ILP 求解开始时间

    x={}  # 当前 partial tree 中动作历史 q -> Gurobi 变量 x_q

    if len(warm_start) != 0: # 复用上一轮变量，仅为新前沿创建变量
        for q,v in warm_start.items():
            x[q] = v  # 复用上一轮已创建的策略变量
        for q in set(frontier + expanded)-set([k for k in warm_start.keys()]):
            x[q] = m.addVar(vtype=gb.GRB.BINARY, name=str(q)) if not continuous else  m.addVar(vtype=gb.GRB.CONTINUOUS, name=str(q))  # 为本轮新增节点创建变量
    else:
        for q in frontier + expanded:
            x[q] = m.addVar(vtype=gb.GRB.BINARY, name=str(q)) if not continuous else m.addVar(vtype=gb.GRB.CONTINUOUS,
                                                                                              name=str(q))  # 首轮为 E/F 中每个动作历史创建变量 x_q
    # 式 (13) 的目标：E 中使用精确局部值，F 中使用可采纳启发式值。
    if len(frontier) != 0 and len(expanded) != 0:
        obj = gb.quicksum([x[q] * T.nodes[q]['u'] if ins.duration_model(q)<ins.horizon else x[q] * T.nodes[q]['h'] for q in expanded]) + gb.quicksum([x[q] * (T.nodes[q]['h']) for q in frontier])  # E 用精确值、F 用启发式值的 p-ILP 目标
    elif len(frontier)!= 0 and len(expanded) == 0:
        obj = gb.quicksum([x[q] * T.nodes[q]['h'] for q in frontier])  # 尚无展开节点时仅由前沿启发式组成的目标
    else:
        obj = gb.quicksum([x[q] * T.nodes[q]['u'] if ins.duration_model(q)<ins.horizon else x[q] * T.nodes[q]['h'] for q in expanded])  # 前沿为空时仅由已展开节点组成的目标

    m.setObjective(obj, gb.GRB.MINIMIZE) if ins.type == "min" else m.setObjective(obj, gb.GRB.MAXIMIZE)


    tree_c1 = gb.quicksum([x[(a,)] for a in np.arange(len(ins.actions))])  # p-ILP 根节点的起始动作选择和
    m.addConstr(tree_c1 == 1, "tree_c1")

    # 只对已展开节点加入后继流约束；前沿节点暂时由启发式封口。
    for q in expanded:
        if ins.duration_model(q) < ins.horizon: # replace with duration model
            for o in np.arange(len(ins.observations)):
                tree_c2 = gb.quicksum([x[q+(o,a)] for a in np.arange(len(ins.actions))])  # 已展开历史 q 在观测 o 后的后继动作选择和
                # print("seq: ", q+(o,))
                m.addConstr(tree_c2 == x[q], "tree_c{}".format(q))
                m.update()

    # 论文要求前沿 F 使用风险下界 h^r_q。当前仓库尚未实现独立的
    # risk_heuristic，而是直接使用节点 r_q，属于论文 HILP 的简化版本。
    if risk:
        if len(frontier) != 0 and len(expanded) != 0:
            capacity_c = gb.quicksum([x[q] * T.nodes[q]['r'] for q in expanded]) + gb.quicksum([x[q] * T.nodes[q]['r'] for q in frontier])  # E 与 F 上的累计风险估计
        elif len(frontier) != 0 and len(expanded) == 0:
            capacity_c = gb.quicksum([x[q] * T.nodes[q]['r'] for q in frontier])  # 初始前沿上的累计风险估计
        else:
            capacity_c = gb.quicksum([x[q] * T.nodes[q]['r'] for q in expanded])  # 已展开策略树上的累计风险

        m.addConstr(capacity_c <= ins.delta)

    m.update()
    m.optimize()

    # return obj.getValue(),{k:v.x for k,v in x.items() if v.x > 0}, time.time()-t1 # only positive results

    # return obj.getValue(),{k:v.X for k,v in x.items()}, time.time()-t1
    return obj.getValue(),x, time.time()-t1

def heuristic_search(ins: ins.Instance, continuous=False, risk=True ):
    """论文 Algorithm 3（HILP）的前向启发式搜索。

    与 preprocess+ILP 一次性建立指数规模的完整树不同，HILP 维护：
      expanded (E)：已展开、使用精确 u_q/r_q 的动作节点；
      frontier (F)：尚未展开、使用 h_q 估计后续价值的边界节点。
    每轮求解 p-ILP，只扩展当前解中 x_q>0 的前沿节点，因此大量不可能进入
    最优策略的历史不会生成。这正是论文实验中 HILP 比完整 ILP 更可扩展的原因。
    """

    total_num_action_vars = sum(  # 完整时域树理论上包含的动作历史/策略变量总数
        [len(ins.actions) ** i * len(ins.observations) ** (i - 1) for i in range(1, ins.horizon + 1)]) + 1
    # print("number of action vars = ", total_num_action_vars - 1)
    size_of_and_or_tree = sum(  # 完整 AND-OR 树理论节点数，仅用于规模统计
        [len(ins.actions) ** i * len(ins.observations) ** (i - 1) for i in range(1, ins.horizon + 1)]) \
                          + sum([len(ins.actions) ** i * len(ins.observations) ** i for i in range(1, ins.horizon)]) + 1
    # print("size of AND-OR tree = ", size_of_and_or_tree)

    m = gb.Model("p_ILP")  # 在多轮搜索中持续增量扩展的 Gurobi 部分 ILP 模型
    m.setParam("OutputFlag", 0)

    t1 = time.perf_counter()  # HILP 搜索开始的高精度计时点

    T = nx.DiGraph()  # HILP 当前已生成的部分 AND-OR 历史树
    T.add_node((), b=ins.b0, rho=1)
    expanded = []  # Algorithm 3 的 E：已展开并使用精确 u_q/r_q 的动作节点
    # Algorithm 3 的初始化：从根开始，只生成四个候选起始动作作为 F。
    frontier = [(a,) for a in np.arange(len(ins.actions))]  # Algorithm 3 的 F：待评估/扩展的边界动作节点
    for q in frontier:
        a = q[0]  # 根节点候选的第一个动作
        T.add_edge((), q)
        b = _update_safe_belief(ins, safe_belief=T.nodes[()]['b'], a=a)  # 首个动作后的安全预测信念
        p_b = T.nodes[()]['b']  # 根节点的初始安全信念 b0
        T.nodes[q]['r'] = sum([b.get(s, 0) * ins.risk_model(s, a) for s in ins.risk_states])  # 根动作节点的局部风险 r_q
        T.nodes[q]['u'] = sum([v * ins.reward_model(s, a) for s, v in p_b.items()])  # 根动作节点的局部期望代价 u_q
        # T.nodes[q]['h'] = sum(
        #     [v * ins.reward_model(s, a) + v * ins.reward_heuristic(s, a) for s, v in b.items()])
        T.nodes[q]['h'] = (1 - ins.delta) * sum([v * ins.reward_heuristic(s, a) for s, v in p_b.items()])  # 根前沿节点的剩余代价启发式 h_q
        T.nodes[q]['b'] = b  # 根动作节点关联的安全预测信念


    loop_count = 0  # HILP 已执行的“求解 p-ILP + 扩展前沿”轮数
    # Algorithm 3 Lines 11--18：求解部分 ILP，并沿当前最优策略扩展前沿。
    sol = {}  # 当前 p-ILP 解：历史 q -> Gurobi 变量对象
    while True:
        # print("iteration {}".format(loop_count))
        # print("frontier size     = {0:7d}".format(len(frontier)), " ({}%)".format(len(frontier)/total_num_action_vars * 100))
        # print("expanded size     = {0:7d}".format(len(expanded)), " ({}%)".format(len(expanded)/total_num_action_vars * 100))
        # print("Total Exploration = {0:7d}".format(len(frontier)+len(expanded)), "({}%)".format((len(frontier)+len(expanded))/total_num_action_vars * 100))
        loop_count+=1

        obj, sol, _ = p_ILP(m,ins,T, expanded=expanded, frontier = frontier, continuous=continuous, risk=risk, warm_start=sol)  # obj：本轮目标值；sol：变量解
        # 只有当前 p-ILP 解真正使用的 frontier 节点才值得继续展开。
        new = [q for q in frontier if sol[q].x > 0]  # 本轮解选中的前沿节点，即接下来真正需要展开的节点
        expanded = expanded + new  # 将已选前沿节点移入精确展开集合 E
        frontier = list(set(frontier) - set(new))  # 从边界集合 F 中移除本轮已展开节点
        for n in new:
            if ins.duration_model(n) < ins.horizon:
                for o in np.arange(len(ins.observations)):
                    # 动作 n 的每种观测都是环境分支（AND），先计算其安全
                    # 后验和历史出现概率，再把所有可选动作加入新前沿。
                    q = n+(o,)  # 在动作历史 n 后观察到 o 的观测历史
                    parent = n  # 当前观测节点的父动作历史
                    T.add_edge(parent, q)
                    grand_parent = [i for i in T.predecessors(parent)][0]  # 上一层观测历史
                    g_rho = T.nodes[grand_parent]['rho']  # 上一层观测历史的安全发生概率
                    b, p = _update_safe_belief(ins, safe_belief=T.nodes[parent]['b'], o=o)  # b：安全后验；p：观测条件概率
                    p_action = n[-1:][0]  # 产生当前观测的父动作编号
                    safe_at = 1 - sum([b.get(s, 0) * ins.risk_model(s, p_action) for s in ins.risk_states])  # 当前历史分支保持安全的条件概率
                    ## TODO require equation modification !!
                    # g_b = T.nodes[grand_parent]['b']
                    # safe_at = 1 - sum([g_b.get(s, 0) * ins.risk_model(s, p_action) for s in ins.risk_states])
                    T.nodes[q]['p'] = p  # 当前观测在安全条件下出现的概率
                    T.nodes[q]['rho'] = g_rho * p * safe_at  # 当前观测历史的安全发生概率 rho~(q)
                    T.nodes[q]['b'] = b  # 当前观测历史对应的安全后验信念

                    for action in np.arange(len(ins.actions)):
                        q = n+(o,action)  # 在观测历史后选择 action 形成的新动作历史
                        parent = q[:-1]  # 新动作节点的父观测历史
                        p_b = T.nodes[parent]['b']  # 新动作执行前的安全后验信念
                        T.add_edge(parent, q)
                        frontier.append(q)
                        p_rho = T.nodes[parent]['rho']  # 父观测历史的安全发生概率
                        b = _update_safe_belief(ins, safe_belief=T.nodes[parent]['b'], a=action)  # 新动作后的安全预测信念
                        T.nodes[q]['r'] = p_rho * sum([b.get(s, 0) * ins.risk_model(s, action) for s in ins.risk_states])  # 新前沿节点的局部风险 r_q
                        T.nodes[q]['u'] = p_rho * sum([v * ins.reward_model(s, action) for s, v in p_b.items()])  # 新前沿节点的局部期望代价 u_q
                        # T.nodes[q]['h'] = p_rho * sum(
                        #     [ v * ins.reward_heuristic(s, action) for s, v in p_b.items()])
                        T.nodes[q]['h'] = p_rho * (1 - ins.delta) * sum(  # 新前沿节点的剩余代价启发式 h_q
                            [v * ins.reward_heuristic(s, action) for s, v in p_b.items()])
                        T.nodes[q]['b'] = b  # 新动作节点关联的安全预测信念



        # 没有被选中的可扩展前沿时，Algorithm 3 的 N 为空，搜索收敛。
        if len(new) == 0:
            break

    print("~~~~~~~~~~~~~~~~~~~~🔥 SUM 🔥~~~~~~~~~~~~~~~~~~~~")
    print("# iteration {}".format(loop_count))
    print("frontier size     = {0:7d}".format(len(frontier)), " ({}%)".format(len(frontier)/total_num_action_vars * 100))
    print("expanded size     = {0:7d}".format(len(expanded)), " ({}%)".format(len(expanded)/total_num_action_vars * 100))
    print("Total Exploration = {0:7d}".format(len(frontier)+len(expanded)), " ({}%)".format((len(frontier)+len(expanded))/total_num_action_vars * 100))
    t2 = time.perf_counter() - t1  # HILP 总运行时间（秒）
    print("Time              = {0:7.3f}".format(t2))
    print("~~~~~~~~~~~~~~~~~~~~🔥~~~~~🔥~~~~~~~~~~~~~~~~~~~~")

    return obj, {k:v.x for k,v in sol.items() if v.x>0}, t2



def _next_belief_states(ins: ins.Instance, states, a):
    """返回当前非零信念状态执行 a 后，一步可达状态的并集。"""

    f = []  # 从当前非零信念支持集一步可达的候选状态列表
    for s in states:
        f = f + [s_ for s_ in ins.trans_model(s, a).keys()]  # 累加状态 s 执行动作 a 后的所有后继状态
    f = set(f)  # 去重后的一步可达状态集合
    f.union(states)
    return f


def _update_belief(ins: ins.Instance, belief, a=None, o=None,normalize=True):
    """普通 POMDP 贝叶斯滤波，对应论文式 (9)--(10)。

    只给 a 时计算预测信念 b'(s')=sum_s b(s)T(s,a,s')；只给 o 时在
    已有预测信念上乘观测似然并归一化；同时给出时完成完整预测和校正。
    主算法为计算 CC-POMDP 执行风险，实际主要使用下面的安全信念版本。
    """

    if a is None and o is None:
        return belief
    prior = {}  # 执行动作后的预测信念 b'(s')
    posterior = {}  # 结合观测后的未归一化/归一化后验信念
    prob_o = 0  # 当前观测 o 的边缘发生概率 Pr(o|b,a)

    if a is None:
        # 观测校正：posterior(s) 正比于 O(o|s,a) * prior(s)。
        for s, v in belief.items():
            posterior[s] = ins.obs_model(s, a)[o] * v  # 状态 s 的未归一化观测后验概率
            prob_o += posterior[s]
        if normalize:
            safe_posterior = {k: v / prob_o for k, v in posterior.items() if v != 0}  # 归一化后的观测后验（变量名沿用旧代码）
        return safe_posterior, prob_o

    next_belief_states = _next_belief_states(ins, belief.keys(), a)  # 当前信念执行动作 a 后可能到达的状态集合

    # print("frontier: ", next_belief_states)
    # print("safe frontier: ", next_safe_belief_states)

    for s in next_belief_states:
        # 只遍历可能到达 s 的前驱状态，执行 Chapman-Kolmogorov 预测。
        S = ins.states_reachable_to[s]  # 一步可能转移到目标状态 s 的前驱状态集合
        val = sum([belief.get(s_, 0) * ins.trans_model(s_, a).get(s, 0) for s_ in S])  # 目标状态 s 的预测概率
        if val != 0:
            prior[s] = val  # 保存非零的预测信念概率
            if o is not None:
                posterior[s] = ins.obs_model(s, a)[o] * prior[s]  # 状态 s 的未归一化动作-观测后验
                prob_o += posterior[s]

    if o is None:
        return prior

    # normalize
    if normalize:
        posterior = {k: v / prob_o for k, v in posterior.items()}  # 用观测概率归一化后的普通后验信念
    return posterior, prob_o


def _update_safe_belief(ins: ins.Instance, safe_belief, a=None, o=None, normalize=True):
    """更新 CC-POMDP 的安全信念，对应论文式 (11) 与 Lemma 3.3。

    安全信念 b~(s) 是“给定此前没有进入风险集合 R”时位于 s 的条件概率。
    动作更新时先用 (1-R(s,a)) 去掉危险前驱的概率质量，再除以 safe_at
    条件化；观测更新与普通贝叶斯滤波相同。该条件分布使整段策略的执行风险
    能被分解为各动作历史的线性项 r_q，从而写入 ILP 风险约束。
    """

    if a is None and o is None:
        return safe_belief
    safe_prior = {}  # 条件于此前未失败时，动作执行后的安全预测信念
    safe_posterior = {}  # 安全预测信念结合观测后的安全后验
    safe_prob_o = 0  # 安全条件下观测 o 的发生概率

    if a is None:
        # 安全预测信念上的观测校正；safe_prob_o 是条件观测概率。
        for s, v in safe_belief.items():
            safe_posterior[s] = ins.obs_model(s, a)[o] * v  # 状态 s 的未归一化安全观测后验
            safe_prob_o += safe_posterior[s]
        if normalize:
            safe_posterior = {k: v / safe_prob_o for k, v in safe_posterior.items() if v != 0}  # 归一化后的安全观测后验
        return safe_posterior, safe_prob_o

    next_safe_belief_states = _next_belief_states(ins, safe_belief.keys(), a)  # 安全信念支持集执行 a 后的可达状态
    # next_safe_belief_states = ins.states
    # 分母 1-r(b~)：在当前安全信念下，本步仍未失败的概率。
    safe_at = 1 - sum([v * ins.risk_model(s, a) for s, v in safe_belief.items()])  # 当前安全信念下本步不失败的概率 1-r(b~)
    for s in next_safe_belief_states:
        # S_ = ins.states_reachable_to[s] - set(ins.risk_states)
        # val_ = sum([safe_belief.get(s_, 0) * ins.trans_model(s_, a).get(s, 0) for s_ in S_]) / safe_at
        # 仅让安全前驱贡献概率，再按 safe_at 归一化，得到安全条件下的
        # 下一状态预测分布 b_q(s)。
        val_ = sum([safe_belief.get(s_, 0) * ins.trans_model(s_, a).get(s, 0) * (1 - ins.risk_model(s_, a)) for s_ in
                    ins.states_reachable_to[s]]) / safe_at  # 下一状态 s 在“本步安全”条件下的预测概率
        if val_ != 0:
            safe_prior[s] = val_  # 保存状态 s 的非零安全预测概率
            if o is not None:
                safe_posterior[s] = ins.obs_model(s, a)[o] * safe_prior[s]  # 状态 s 的未归一化安全动作-观测后验
                safe_prob_o += safe_posterior[s]

    if o is None:
        return safe_prior

    if normalize:
        safe_posterior = {k: v / safe_prob_o for k, v in safe_posterior.items()}  # 用安全观测概率归一化后的安全后验
    return safe_posterior, safe_prob_o

##################################################
#### 以下函数只用于调试和实验，不属于论文 Algorithm 1--3 主流程 ####
##################################################


# returns <a_1,o_1,a_2,o_2,...,a_t,o_t>
def BFS(ins: ins.Instance):
    # yield ()
    for i in np.arange(0, ins.horizon + 1):
        for seq in itertools.product(np.arange(len(ins.actions)), np.arange(len(ins.observations)), repeat=i):  # depth first search
            yield seq


# returns <a_1,o_1,a_2,o_2,...,o_t,a_t+1>
def BFS_(ins: ins.Instance):
    for a in np.arange(len(ins.actions)):
        yield (a,)
    for a in np.arange(len(ins.actions)):
        for i in np.arange(1, ins.horizon):
            for seq in itertools.product(np.arange(len(ins.observations)), np.arange(len(ins.actions)), repeat=i):  # depth first search
                q = (a,) + seq  # 由首动作和后续观测-动作对组成的调试历史
                yield q

# assume a_seq and o_seq have the same size
def history(ins: ins.Instance, current_belief=None, current_safe_belief=None, a_seq=[0], o_seq=None):
    if current_belief is None:
        current_belief = ins.b0  # 普通信念更新的起点，默认为初始信念 b0
    if current_safe_belief is None:
        current_safe_belief = ins.b0  # 安全信念更新的起点，默认假设 b0 完全安全

    for i in range(len(a_seq)):
        if o_seq is None:
            b = _update_belief(ins, belief=current_belief, a=a_seq[i])  # 第 i 个动作后的普通预测信念
            b_ = _update_safe_belief(ins, safe_belief=current_safe_belief, a=a_seq[i])  # 第 i 个动作后的安全预测信念
        else:
            b, p = _update_belief(ins, belief=current_belief, a=a_seq[i], o=o_seq[i])  # 普通后验 b 及观测概率 p
            b_, p_ = _update_safe_belief(ins, safe_belief=current_safe_belief, a=a_seq[i], o=o_seq[i])  # 安全后验 b_ 及安全观测概率 p_
            print("p_ = ", p_)
            print("Action {} Observation {}".format(ins.action_to_string(a_seq[i]), o_seq[i]))
        current_belief = b  # 将普通后验作为下一步普通信念的输入
        current_safe_belief = b_  # 将安全后验作为下一步安全信念的输入
        # print("posterior      = ", current_belief)
        # print("p = ", p)
        print("safe posterior = ", current_safe_belief)
        print()
