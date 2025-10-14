# my_planner.py
import numpy as np
import math
import time
import random
from collections import defaultdict


# ------------------------------
# MCTS Node
# ------------------------------
class Node:
    """MCTS 搜索树节点（reward_from_parent 存边回报）"""
    def __init__(self, state, parent=None, action=None, reward=0.0):
        self.state = np.array(state, dtype=np.int32)
        self.parent = parent
        self.action = action
        self.reward_from_parent = reward
        self.children = {}          # { action_int: Node }
        self.visits = 0
        self.value = 0.0            # 累计折扣回报之和（均值用于 UCT）
        self.untried_actions = None
        self.is_terminal = False
        self.depth = parent.depth + 1 if parent is not None else 0

    def q_value(self) -> float:
        return 0.0 if self.visits == 0 else (self.value / self.visits)

    def is_fully_expanded(self) -> bool:
        return not self.untried_actions


# ===============================
# MyPlanner（含拥堵感知 + 安全增益 + 侧向偏置）
# 自适应可选，接口与原版兼容
# ===============================
class MyPlanner:
    def __init__(self, env,
                 # —— 基础 —— #
                 exploration_constant: float | None = 3.0,
                 rollout_epsilon: float = 0.2,
                 max_rollout_depth: int | None = 20,
                 # —— APW / 先验 —— #
                 apw_k: float = 10.0,
                 apw_alpha: float = 0.5,
                 topk_per_drone: int = 5,
                 max_prior_samples: int = 100,
                 # —— 终局模式 —— #
                 end_active_threshold: int = 1,   # 默认提前到1：仅剩1台未达即触发终局策略
                 end_rollout_depth: int = 60,
                 end_epsilon: float = 0.1,
                 end_c_scale: float = 0.4,
                 heuristic_alpha: float = -5.0,
                 stay_action_index: int = 0,
                 # —— 安全增益 / 侧向偏置 —— #
                 use_clearance_gain: bool = True,
                 use_lateral_bias: bool = True,
                 asym_tau: float = 0.12,   # 地图非对称度阈值，小于它时视为对称（两项增强基本熄火）
                 # —— 自适应 —— #
                 auto_param: bool = True,
                 auto_adjust_period: int = 20,
                 adapt_log: bool = False):
        self.env = env
        cfg = env.get_config()

        # Global config
        self.discount_factor = cfg.discount_factor
        self.max_steps_global = cfg.max_num_steps

        # Determine c_max based on action space size (adaptive limits)
        num_actions = env.num_actions
        if num_actions < 100:
            self.c_max = 30.0
        elif num_actions < 1000:
            self.c_max = 60.0
        elif num_actions < 100000:
            self.c_max = 120.0
        else:
            self.c_max = 150.0

        # Base parameters
        if exploration_constant is None:
            self.c = 50.0 if num_actions > 1000 else 10.0
        else:
            self.c = exploration_constant
        self.c = float(np.clip(self.c, 0.01, self.c_max))

        self.epsilon = rollout_epsilon
        self.max_rollout_depth = min(
            max_rollout_depth if max_rollout_depth is not None else 30,
            self.max_steps_global
        )
        self.apw_k = apw_k
        self.apw_alpha = apw_alpha
        self.topk_per_drone = topk_per_drone
        self.max_prior_samples = max_prior_samples

        # 终局
        self.end_active_threshold = end_active_threshold
        self.end_rollout_depth = min(end_rollout_depth, self.max_steps_global)
        self.end_epsilon = end_epsilon
        self.end_c_scale = end_c_scale
        self.heuristic_alpha = heuristic_alpha
        self.stay_action_index = stay_action_index

        # 环境静态
        self.goals = np.array(cfg.goal_positions)
        self.action_vectors = self.env._action_vectors      # (A,3)
        self.num_per_drone_actions = self.action_vectors.shape[0]

        # 树重用
        self.root: Node | None = None

        # 自适应
        self.auto_param = auto_param
        self.auto_adjust_period = max(5, int(auto_adjust_period))
        self._profiled = False
        self._last_h = None
        self._stale = 0
        self._adapt_log = adapt_log

        # 安全增益 / 侧向偏置 开关与阈值
        self.use_clearance_gain = use_clearance_gain
        self.use_lateral_bias = use_lateral_bias
        self._asym_tau = max(0.0, float(asym_tau))

        # —— 地图侧向偏置（一次性预计算）——
        self._lateral_bias_vec, self._lateral_asym = self._compute_clearance_axis_bias()

    # ==================== 主入口 ====================
    def plan(self, current_state: np.ndarray,
             planning_time_per_step: float | None = None,
             num_iterations: int | None = None) -> int:

        # （可选）地图画像 → 初值
        if self.auto_param and not self._profiled:
            prof = self._profile_map(current_state)
            self._apply_map_presets(prof)
            self._profiled = True
            if self._adapt_log:
                print(f"[ADAPT:init] Actions={self.env.num_actions} c_max={self.c_max:.1f} "
                      f"c={self.c:.2f} eps={self.epsilon:.2f} depth={self.max_rollout_depth} "
                      f"topk={self.topk_per_drone} prior={self.max_prior_samples} apw_alpha={self.apw_alpha:.2f}")

        # 提前判终
        _, _, root_done, _ = self.env.simulate(current_state, self.stay_action_index)
        if root_done:
            return self.stay_action_index

        # 树重用
        if self.root is not None:
            matched = None
            for child in self.root.children.values():
                if np.array_equal(child.state, current_state):
                    matched = child
                    break
            if matched is not None:
                self.root = matched
                self.root.parent = None
            else:
                self.root = Node(state=current_state)
                self.root.untried_actions = self._get_prioritized_actions(current_state)
        else:
            self.root = Node(state=current_state)
            self.root.untried_actions = self._get_prioritized_actions(current_state)

        # 预算循环
        start = time.time()
        iters = 0
        use_time = planning_time_per_step is not None

        while True:
            if use_time:
                if time.time() - start > planning_time_per_step:
                    break
            else:
                if num_iterations is not None and iters >= num_iterations:
                    break

            self._run_mcts_iteration(self.root)
            iters += 1

            # 在线微调（极低开销）
            if self.auto_param and (iters % self.auto_adjust_period == 0):
                self._online_adjust(self.root)
                if self._adapt_log:
                    print(f"[ADAPT:step] it={iters} c={self.c:.4f} eps={self.epsilon:.3f} "
                          f"depth={self.max_rollout_depth}")
                if self.c > 12.0:
                    print(f"[WARN] c={self.c:.4f} approaching limit, consider investigating")

        if not self.root.children:
            return self._get_greedy_action(current_state)

        # 根下选访问次数最大
        best_action = max(self.root.children, key=lambda a: self.root.children[a].visits)
        return best_action

    # ==================== MCTS 四步 ====================
    def _run_mcts_iteration(self, root: Node):
        leaf = self._select_and_expand(root)
        rollout_ret = self._rollout(leaf.state, leaf.depth)
        self._backpropagate(leaf, rollout_ret)

    def _select_and_expand(self, node: Node) -> Node:
        cur = node
        while not cur.is_terminal:
            # APW：限制开放孩子数量随访问数次幂增长（更稳：取整 & visits>=1）
            limit = self.apw_k * (max(1, cur.visits) ** self.apw_alpha)
            if len(cur.children) < int(limit):
                if not cur.is_fully_expanded():
                    a = cur.untried_actions.pop(0)
                    ns, r, done, _ = self.env.simulate(cur.state, a)
                    child = Node(state=ns, parent=cur, action=a, reward=r)
                    child.is_terminal = done
                    if not done:
                        child.untried_actions = self._get_prioritized_actions(ns)
                    cur.children[a] = child
                    return child

            if not cur.children:
                return cur

            cur = self._best_child_uct(cur)
        return cur

    def _rollout(self, state: np.ndarray, depth: int) -> float:
        """终局自适应（更深、更贪心），截断用启发式兜底，并在拥堵时降低随机度"""
        total = 0.0
        s = np.copy(state)
        gamma = self.discount_factor
        endgame = self._is_endgame(s)

        local_eps = self.end_epsilon if endgame else self.epsilon
        carry = self.end_rollout_depth if endgame else self.max_rollout_depth

        # 拥堵检测：邻近(<=1)对数过多 -> 降低随机度，避免反复扰动导致“再拥堵”
        pos = s[:, :3]
        crowd = 0
        for i in range(len(pos)):
            crowd += np.sum(np.max(np.abs(pos - pos[i]), axis=1) <= 1) - 1
        if crowd >= 4:
            local_eps = max(0.02, local_eps * 0.5)

        for i in range(carry):
            cur_depth = depth + i
            if cur_depth >= self.max_steps_global:
                break
            a = self._rollout_policy(
                s,
                greedy=True if endgame else False,
                epsilon_override=local_eps
            )
            ns, r, done, _ = self.env.simulate(s, a)

            # 安全累积奖励
            discount = gamma ** i
            total += discount * r

            if np.isnan(total) or np.isinf(total):
                return 0.0  # 返回中性值

            s = ns
            if done:
                return total

        # 截断：启发式兜底（加入轻微的“分离”项，避免拥挤状态被高估）
        heuristic = self._heuristic_value(s)
        discount = gamma ** carry
        total += discount * heuristic

        if np.isnan(total) or np.isinf(total):
            return 0.0

        return total

    def _backpropagate(self, node: Node, rollout_return: float):
        """回传一份标量：沿父链做 G_{t-1} = r_{t-1} + γ G_t"""
        cur = node
        G = rollout_return
        gamma = self.discount_factor
        while cur is not None:
            cur.visits += 1
            cur.value += G
            G = cur.reward_from_parent + gamma * G
            cur = cur.parent

    # ==================== 组件：UCT/先验 ====================
    def _best_child_uct(self, node: Node) -> Node:
        # 终局时降低探索常数（偏利用）
        c_use = self.c * (self.end_c_scale if self._is_endgame(node.state) else 1.0)

        logN = math.log(max(node.visits, 1) + 1.0)
        best, best_score = None, -float('inf')

        for child in node.children.values():
            exploit = child.q_value()
            visits_safe = max(child.visits, 0) + 1e-6
            explore_term = logN / visits_safe
            explore_term = max(explore_term, 0.0)
            explore = c_use * math.sqrt(explore_term)
            score = exploit + explore
            if np.isnan(score) or np.isinf(score):
                continue
            if score > best_score:
                best_score, best = score, child

        if best is None and node.children:
            best = max(node.children.values(), key=lambda ch: ch.visits)
        return best

    def _get_greedy_action(self, state: np.ndarray) -> int:
        return self._rollout_policy(state, greedy=True)

    # === 拥堵感知先验：确保“stay”始终可选，并按照评分器排序 ===
    def _get_prioritized_actions(self, state: np.ndarray) -> list[int]:
        num_drones = state.shape[0]
        endgame = self._is_endgame(state)
        k = min(self.topk_per_drone, self.num_per_drone_actions if not endgame else 3)

        per_top: list[list[int]] = []
        for i in range(num_drones):
            if state[i, 3] == 1:  # 已到达 -> 只允许停留
                per_top.append([self.stay_action_index])
                continue
            scores = [(a, self._score_move(i, state, a)) for a in range(self.num_per_drone_actions)]
            if self.stay_action_index not in [a for a, _ in scores]:
                scores.append((self.stay_action_index, self._score_move(i, state, self.stay_action_index)))
            scores.sort(key=lambda x: x[1], reverse=True)
            per_top.append([a for a, _ in scores[:max(k, 1)]])

        # 组合：先最佳，再采样
        best_combo = [tops[0] for tops in per_top]
        prioritized: list[int] = []
        prioritized.append(self.env._encode_action(np.array(best_combo, dtype=np.int32)))

        cap = min(self.max_prior_samples if not endgame else max(30, self.max_prior_samples // 2),
                  self.env.num_actions)
        seen = {prioritized[0]}
        tries = 0
        while len(prioritized) < cap and tries < cap * 3:
            combo = [random.choice(tops) for tops in per_top]
            a = self.env._encode_action(np.array(combo, dtype=np.int32))
            if a not in seen:
                prioritized.append(a)
                seen.add(a)
            tries += 1
        return prioritized

    # === Rollout 同样用拥堵感知评分器（保留冲突消解） ===
    def _rollout_policy(self, state: np.ndarray,
                        greedy: bool = False,
                        epsilon_override: float | None = None) -> int:
        eps = self.epsilon if epsilon_override is None else epsilon_override
        if not greedy and random.random() < eps:
            return random.randint(0, self.env.num_actions - 1)

        num_drones = state.shape[0]
        per_sorted = []
        active_mask, reached = self._active_drones_mask(state)

        for i in range(num_drones):
            if reached[i]:
                per_sorted.append([self.stay_action_index])
                continue
            all_scores = [(a, self._score_move(i, state, a)) for a in range(self.num_per_drone_actions)]
            if self.stay_action_index not in [a for a, _ in all_scores]:
                all_scores.append((self.stay_action_index, self._score_move(i, state, self.stay_action_index)))
            all_scores.sort(key=lambda x: x[1], reverse=True)
            per_sorted.append([a for a, _ in all_scores])

        # 冲突解冲：从最优开始，冲突方后移
        p_idx = np.zeros(num_drones, dtype=int)
        for _ in range(num_drones + 1):
            intended = np.zeros(num_drones, dtype=int)
            targets = defaultdict(list)

            for i in range(num_drones):
                a_i = per_sorted[i][min(p_idx[i], len(per_sorted[i]) - 1)]
                if a_i == self.stay_action_index:
                    tpos = tuple(state[i, :3])
                else:
                    tpos = tuple((state[i, :3] + self.action_vectors[a_i]).tolist())
                intended[i] = a_i
                targets[tpos].append(i)

            losers = {
                d for drones in targets.values() if len(drones) > 1
                for d in drones if d != min(drones)
            }
            if not losers:
                return self.env._encode_action(intended.astype(int))

            for i in losers:
                if p_idx[i] + 1 < len(per_sorted[i]):
                    p_idx[i] += 1
                else:
                    p_idx[i] = random.randint(0, len(per_sorted[i]) - 1)

        # 仍冲突则全局随机
        return random.randint(0, self.env.num_actions - 1)

    # ==================== 工具/启发 ====================
    def _active_drones_mask(self, state: np.ndarray):
        reached = state[:, 3].astype(bool)
        active_mask = ~reached
        return active_mask, reached

    def _is_endgame(self, state: np.ndarray) -> bool:
        active_mask, _ = self._active_drones_mask(state)
        return int(active_mask.sum()) <= self.end_active_threshold

    def _heuristic_value(self, state: np.ndarray) -> float:
        """距离启发 + 轻量分离项：负的未达无人机到目标的曼哈顿距离和 + 0.1*pairwise分离"""
        active_mask, _ = self._active_drones_mask(state)
        if not active_mask.any():
            return 0.0
        pos = state[:, :3]
        d_goal = np.sum(np.abs(pos[active_mask] - self.goals[active_mask]), axis=1).sum()

        sep = 0.0
        P = pos[active_mask]
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                sep += np.max(np.abs(P[i] - P[j]))
        return self.heuristic_alpha * float(d_goal) + 0.1 * float(sep)

    # ==================== 安全增益 + 侧向偏置（一次性画像 & 评分器内融合） ====================
    def _compute_clearance_axis_bias(self):
        """
        基于距离场的简单“左右（或前后）安全度”统计，返回一个单位向量方向偏置（3D 下 z 分量为 0）。
        若左半场平均 clearance << 右半场，就鼓励动作的 x 分量朝 +x（右）微偏，反之亦然。
        """
        df = self.env.dist_field  # (X,Y,Z)
        if df.ndim == 2:
            df_xy = df
            X, Y = df.shape
            Z = 1
        else:
            X, Y, Z = df.shape
            df_xy = df.mean(axis=2) if Z > 1 else df[..., 0]

        mid = X // 2
        left_mean  = float(df_xy[:mid, :].mean()) if mid > 0 else 0.0
        right_mean = float(df_xy[mid:, :].mean()) if (X - mid) > 0 else 0.0

        eps = 1e-6
        asym = abs(right_mean - left_mean) / (((left_mean + right_mean) / 2.0) + eps)

        if right_mean > left_mean:
            bias = np.array([1.0, 0.0, 0.0], dtype=float)  # 朝 +x
        elif right_mean < left_mean:
            bias = np.array([-1.0, 0.0, 0.0], dtype=float) # 朝 -x
        else:
            bias = np.zeros(3, dtype=float)

        n = np.linalg.norm(bias)
        if n > 0:
            bias = bias / n

        return bias, asym

    def _score_move(self, i: int, state: np.ndarray, action_idx: int) -> float:
        """单机动作评分：越大越好（兼顾朝目标、离障碍、与他机分离、垂向分流 + 安全增益 + 侧向偏置）"""
        pos_i = state[i, :3]
        tgt = pos_i if action_idx == self.stay_action_index else (pos_i + self.action_vectors[action_idx])
        tgt = tgt.astype(int)

        # 1) 目标进展
        g = self.goals[i]
        goal_gain = -np.sum(np.abs(tgt - g))

        # 2) 目标格安全度
        X, Y, Z = self.env.cfg.grid_size
        tx, ty, tz = np.clip(tgt, [0, 0, 0], [X - 1, Y - 1, Z - 1])
        obs_clear = float(self.env.dist_field[tx, ty, tz])

        # 2.5) 安全“增益”：目标格 - 当前格（>0 表示离障碍更远）
        cx, cy, cz = np.clip(pos_i, [0, 0, 0], [X - 1, Y - 1, Z - 1])
        cur_clear = float(self.env.dist_field[int(cx), int(cy), int(cz)])
        clearance_gain = obs_clear - cur_clear

        # 3) 与他机分离（Chebyshev 最小距离）
        others = np.delete(state[:, :3], i, axis=0)
        if others.size:
            sep = np.min(np.max(np.abs(others - tgt), axis=1))
        else:
            sep = 3.0

        # 4) 3D 时的轻微垂向偏好（帮助穿过瓶颈）
        vertical = 1.0 if (self.env.cfg.change_altitude and self.action_vectors[action_idx][2] != 0) else 0.0

        # 5) 侧向偏置（仅水平分量参与）
        move_vec = (tgt - pos_i).astype(float)
        move_vec[2] = 0.0
        lateral_proj = 0.0
        if np.linalg.norm(move_vec) > 0 and np.linalg.norm(self._lateral_bias_vec) > 0:
            lateral_proj = float(np.dot(move_vec / (np.linalg.norm(move_vec) + 1e-9),
                                        self._lateral_bias_vec))

        # —— 权重（根据非对称度稍作自适应放大）——
        # 原有权重
        w_goal, w_clear, w_sep, w_vert = 1.0, 0.6, 0.5, 0.25
        # 新权重（基线）
        base_w_gain, base_w_lat = 0.35, 0.20
        # 地图非对称度（上限1.5）→ 放大安全相关与侧向（温和）
        asym_k = min(self._lateral_asym, 1.5)
        w_gain = base_w_gain * (1.0 + 0.6 * asym_k)
        w_lat  = base_w_lat  * (1.0 + 0.6 * asym_k)
        w_clear_eff = w_clear * (1.0 + 0.4 * asym_k)

        # 阈值熔断与开关：对“近似对称”的图，增强项基本关闭
        if (not self.use_clearance_gain) or (self._lateral_asym < self._asym_tau):
            w_gain = 0.0
        if (not self.use_lateral_bias) or (self._lateral_asym < self._asym_tau):
            w_lat = 0.0

        return (w_goal * goal_gain
                + w_clear_eff * obs_clear
                + w_gain * clearance_gain
                + w_sep * sep
                + w_vert * vertical
                + w_lat * lateral_proj)

    # ==================== 自适应：地图画像 & 在线微调（可开可关） ====================
    def _profile_map(self, state: np.ndarray) -> dict:
        rho = 0.5
        if hasattr(self.env, "obstacle_density"):
            try:
                rho = float(self.env.obstacle_density)
            except Exception:
                pass

        pos = state[:, :3]
        dbar = float(np.mean(np.sum(np.abs(pos - self.goals), axis=1)))
        A_prior = min(self.max_prior_samples, getattr(self.env, "num_actions", 200))
        return {"rho": max(0.0, min(1.0, rho)), "dbar": dbar, "Aprior": A_prior}

    def _apply_map_presets(self, prof: dict):
        rho, dbar, A = prof["rho"], prof["dbar"], prof["Aprior"]
        log_term = 0 if A <= 10 else min(math.log10(max(A, 1)), 3.0)
        base_c = 6.0 + 4.0 * log_term
        self.c = base_c * (1.0 + 0.5 * rho)
        self.c = float(np.clip(self.c, 0.5, self.c_max))

        self.epsilon = max(0.05, min(0.05 + 0.15 * rho, 0.25))
        self.max_rollout_depth = int(max(15, min(round(0.8 * dbar) + (10 if rho > 0.5 else 0), 40)))
        self.topk_per_drone = max(4, min(4 + int(round(2 * rho)), 8))
        self.max_prior_samples = int(max(80, min(100 + 80 * rho + 0.1 * A, 220)))
        self.apw_alpha = max(0.5, min(0.5 + 0.2 * rho, 0.8))
        self.apw_k = 8

    def _online_adjust(self, root: "Node"):
        totalN = sum(ch.visits for ch in root.children.values()) + 1e-9
        maxN = max((ch.visits for ch in root.children.values()), default=0)
        p_star = maxN / totalN

        if p_star < 0.35:
            self.c += np.tanh(0.15)
        elif p_star > 0.60:
            self.c += -np.tanh(0.10)
        self.c = float(np.clip(self.c, 0.01, self.c_max))
        if np.isnan(self.c) or np.isinf(self.c):
            if self._adapt_log:
                print(f"[WARN] c overflow detected, resetting to 3.0")
            self.c = 3.0

        cur_h = self._heuristic_value(root.state)
        if self._last_h is None:
            self._last_h = cur_h
        delta = cur_h - self._last_h
        self._last_h = cur_h

        if delta < 0.5:
            self._stale += 1
        else:
            self._stale = 0

        if self._stale >= 3:
            self.epsilon = max(0.03, self.epsilon - 0.03)
            self.max_rollout_depth = min(self.max_rollout_depth + 5, self.end_rollout_depth)
            self._stale = 0
        elif delta > 1.0:
            self.epsilon = min(0.20, self.epsilon + 0.02)


# ------------------------------
# 命令行参数接入（保持你的原接口，新增了增强项的开关与阈值）
# ------------------------------
def add_planner_args(parser):
    p = parser
    # 基础
    p.add_argument("--exploration_constant", type=float, default=3.0)
    p.add_argument("--rollout_epsilon", type=float, default=0.2)
    p.add_argument("--max_rollout_depth", type=int, default=20)
    # 先验/APW
    p.add_argument("--apw_k", type=float, default=10.0)
    p.add_argument("--apw_alpha", type=float, default=0.5)
    p.add_argument("--topk_per_drone", type=int, default=5)
    p.add_argument("--max_prior_samples", type=int, default=100)
    # 终局
    p.add_argument("--end_active_threshold", type=int, default=1)
    p.add_argument("--end_rollout_depth", type=int, default=60)
    p.add_argument("--end_epsilon", type=float, default=0.1)
    p.add_argument("--end_c_scale", type=float, default=0.4)
    p.add_argument("--heuristic_alpha", type=float, default=-5.0)
    p.add_argument("--stay_action_index", type=int, default=0)
    # 安全增益 / 侧向偏置
    p.add_argument("--use_clearance_gain", action="store_true", default=True)
    p.add_argument("--no_use_clearance_gain", dest="use_clearance_gain", action="store_false")
    p.add_argument("--use_lateral_bias", action="store_true", default=True)
    p.add_argument("--no_use_lateral_bias", dest="use_lateral_bias", action="store_false")
    p.add_argument("--asym_tau", type=float, default=0.12)
    # 自适应
    p.add_argument("--auto_param", action="store_true", default=True)
    p.add_argument("--no_auto_param", dest="auto_param", action="store_false")
    p.add_argument("--auto_adjust_period", type=int, default=20)
    p.add_argument("--adapt_log", action="store_true", default=False)


def make_planner_from_args(env, args) -> MyPlanner:
    return MyPlanner(
        env=env,
        exploration_constant=args.exploration_constant,
        rollout_epsilon=args.rollout_epsilon,
        max_rollout_depth=args.max_rollout_depth,
        apw_k=args.apw_k,
        apw_alpha=args.apw_alpha,
        topk_per_drone=args.topk_per_drone,
        max_prior_samples=args.max_prior_samples,
        end_active_threshold=args.end_active_threshold,
        end_rollout_depth=args.end_rollout_depth,
        end_epsilon=args.end_epsilon,
        end_c_scale=args.end_c_scale,
        heuristic_alpha=args.heuristic_alpha,
        stay_action_index=args.stay_action_index,
        use_clearance_gain=args.use_clearance_gain,
        use_lateral_bias=args.use_lateral_bias,
        asym_tau=args.asym_tau,
        auto_param=args.auto_param,
        auto_adjust_period=args.auto_adjust_period,
        adapt_log=args.adapt_log,
    )
