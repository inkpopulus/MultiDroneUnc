import numpy as np
import math
import time
import random
from collections import defaultdict
from multi_drone import MultiDroneUnc


# ------------------------------
# MCTS Node
# ------------------------------
class Node:
    """MCTS search tree node"""
    def __init__(self, state, parent=None, action=None, reward=0.0):
        self.state = np.array(state, dtype=np.int32)
        self.parent = parent
        self.action = action
        self.reward_from_parent = reward
        self.children = {}          # { action_int: Node }
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None
        self.is_terminal = False
        self.depth = parent.depth + 1 if parent is not None else 0

    def q_value(self) -> float:
        return 0.0 if self.visits == 0 else (self.value / self.visits)

    def is_fully_expanded(self) -> bool:
        return not self.untried_actions


# ------------------------------
# MCTS Planner
# ------------------------------
class MyPlanner:
    """
    Adaptive multi-drone UCT-MCTS with symmetry breaking and swap risk penalty
    """
    def __init__(self, env: MultiDroneUnc,
                 # Basic parameters
                 exploration_constant: float | None = 3.0,
                 rollout_epsilon: float = 0.2,
                 max_rollout_depth: int | None = 20,
                 # APW / Prior
                 apw_k: float = 10.0,
                 apw_alpha: float = 0.5,
                 topk_per_drone: int = 5,
                 max_prior_samples: int = 100,
                 # Endgame mode
                 end_active_threshold: int = 2,
                 end_rollout_depth: int = 60,
                 end_epsilon: float = 0.1,
                 end_c_scale: float = 0.4,
                 heuristic_alpha: float = -5.0,
                 stay_action_index: int = 0,
                 # Optional CLI-compatible parameters (stored but not enforced)
                 use_clearance_gain: bool = True,
                 use_lateral_bias: bool = True,
                 asym_tau: float = 0.12,
                 # Adaptive parameters
                 auto_param: bool = True,
                 auto_adjust_period: int = 20,
                 adapt_log: bool = False):
        self.env = env
        cfg = env.get_config()

        # Global config
        self.discount_factor = cfg.discount_factor
        self.max_steps_global = cfg.max_num_steps

        # Environment static info
        self.goals = np.array(cfg.goal_positions)
        self.action_vectors = self.env._action_vectors      # (A,3)
        self.num_per_drone_actions = self.action_vectors.shape[0]

        # Determine c_max based on action space size
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
        self.max_rollout_depth = min(max_rollout_depth if max_rollout_depth is not None else 30,
                                     self.max_steps_global)
        self.apw_k = apw_k
        self.apw_alpha = apw_alpha
        self.topk_per_drone = topk_per_drone
        self.max_prior_samples = max_prior_samples

        # Endgame parameters
        self.end_active_threshold = end_active_threshold
        self.end_rollout_depth = min(end_rollout_depth, self.max_steps_global)
        self.end_epsilon = end_epsilon
        self.end_c_scale = end_c_scale
        self.heuristic_alpha = heuristic_alpha
        self.stay_action_index = stay_action_index

        # Optional CLI-compatible parameters (stored only, not enforced in this version)
        self.use_clearance_gain = use_clearance_gain
        self.use_lateral_bias = use_lateral_bias
        self._asym_tau = asym_tau

        # Tree reuse and adaptive tuning
        self.root: Node | None = None
        self.auto_param = auto_param
        self.auto_adjust_period = max(5, int(auto_adjust_period))
        self._profiled = False
        self._last_h = None
        self._stale = 0
        self._adapt_log = adapt_log

    # ==================== Main entry ====================
    def plan(self, current_state: np.ndarray,
             planning_time_per_step: float | None = None,
             num_iterations: int | None = None) -> int:

        # Optional: map profiling for initial parameter adjustment
        if self.auto_param and not self._profiled:
            prof = self._profile_map(current_state)
            self._apply_map_presets(prof)
            self._profiled = True
            if self._adapt_log:
                print(f"[ADAPT:init] c={self.c:.2f} eps={self.epsilon:.2f} depth={self.max_rollout_depth} "
                      f"topk={self.topk_per_drone} prior={self.max_prior_samples}")

        # Early termination check
        _, _, root_done, _ = self.env.simulate(current_state, self.stay_action_index)
        if root_done:
            return self.stay_action_index

        # Tree reuse
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

        # Planning budget loop
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

            # Online parameter adjustment
            if self.auto_param and (iters % self.auto_adjust_period == 0):
                self._online_adjust(self.root)
                if self._adapt_log:
                    print(f"[ADAPT:step] it={iters} c={self.c:.4f} eps={self.epsilon:.3f} "
                          f"depth={self.max_rollout_depth}")
                if self.c > 12.0:
                    print(f"[WARN] c={self.c:.4f} approaching limit")

        if not self.root.children:
            return self._get_greedy_action(current_state)

        # Select action with most visits at root
        best_action = max(self.root.children, key=lambda a: self.root.children[a].visits)
        return best_action

    # ==================== MCTS four phases ====================
    def _run_mcts_iteration(self, root: Node):
        leaf = self._select_and_expand(root)
        rollout_ret = self._rollout(leaf.state, leaf.depth)
        self._backpropagate(leaf, rollout_ret)

    def _select_and_expand(self, node: Node) -> Node:
        cur = node
        while not cur.is_terminal:
            # APW: limit children expansion based on visit count
            if len(cur.children) < self.apw_k * (cur.visits ** self.apw_alpha):
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
        """Endgame-adaptive rollout with deeper/greedier exploration, heuristic backup for truncation"""
        total = 0.0
        s = np.copy(state)
        gamma = self.discount_factor
        endgame = self._is_endgame(s)

        local_eps = self.end_epsilon if endgame else self.epsilon
        carry = self.end_rollout_depth if endgame else self.max_rollout_depth

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

            total += (gamma ** i) * r
            if np.isnan(total) or np.isinf(total):
                return 0.0

            s = ns
            if done:
                return total

        heuristic = self._heuristic_value(s)
        total += (gamma ** carry) * heuristic
        if np.isnan(total) or np.isinf(total):
            return 0.0
        return total

    def _backpropagate(self, node: Node, rollout_return: float):
        cur = node
        G = rollout_return
        gamma = self.discount_factor
        while cur is not None:
            cur.visits += 1
            cur.value += G
            G = cur.reward_from_parent + gamma * G
            cur = cur.parent

    # ==================== Components: UCT/Prior ====================
    def _best_child_uct(self, node: Node) -> Node:
        c_use = self.c * (self.end_c_scale if self._is_endgame(node.state) else 1.0)
        logN = math.log(max(node.visits, 1) + 1.0)
        best, best_score = None, -float('inf')

        for child in node.children.values():
            exploit = child.q_value()
            explore_term = max(logN / (child.visits + 1e-6), 0.0)
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

    def _get_prioritized_actions(self, state: np.ndarray) -> list[int]:
        """
        Heuristic action prioritization with obstacle awareness and symmetry breaking:
        - Endgame: freeze reached drones to stay action (drastically reduces joint action space)
        - Otherwise: take top-k actions per drone based on goal proximity, obstacle safety, and penalty reduction
        """
        num_drones = state.shape[0]
        pos = state[:, :3]
        _, reached = self._active_drones_mask(state)
        endgame = self._is_endgame(state)

        k = min(self.topk_per_drone, self.num_per_drone_actions if not endgame else 2)
        per_top: list[list[int]] = []
        X, Y, Z = self.env.cfg.grid_size

        for i in range(num_drones):
            if reached[i]:
                per_top.append([self.stay_action_index])
            else:
                base_dist = np.sum(np.abs(pos[i] + self.action_vectors - self.goals[i]), axis=1)
                adj = base_dist.astype(float).copy()
                for a in range(self.num_per_drone_actions):
                    tgt = pos[i] if a == self.stay_action_index else (pos[i] + self.action_vectors[a])

                    # Obstacle penalty
                    tx, ty, tz = np.clip(tgt, [0, 0, 0], [X-1, Y-1, Z-1]).astype(int)
                    obs_dist = float(self.env.dist_field[tx, ty, tz])
                    obstacle_penalty = max(0, 3.0 - obs_dist) * 3.0

                    # Symmetry breaking and swap risk penalties
                    sym_pen = self._right_of_way_penalty(i, state, tgt)
                    swap_pen = self._swap_risk_penalty(i, state, tgt)

                    # Combine all penalties
                    adj[a] += obstacle_penalty - (sym_pen + swap_pen)
                # Select top-k actions
                idx = np.argsort(adj)
                per_top.append(idx[:k].tolist())

        # Combine: best first, then sample
        prioritized: list[int] = []
        best_combo = [tops[0] for tops in per_top]
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

    # ==================== Components: Rollout policy ====================
    def _rollout_policy(self, state: np.ndarray,
                        greedy: bool = False,
                        epsilon_override: float | None = None) -> int:
        eps = self.epsilon if epsilon_override is None else epsilon_override
        if not greedy and random.random() < eps:
            return random.randint(0, self.env.num_actions - 1)

        num_drones = state.shape[0]
        per_sorted = []
        active_mask, reached = self._active_drones_mask(state)

        # Per-drone sorting by goal distance + obstacle penalty + symmetry breaking/swap risk penalties
        X, Y, Z = self.env.cfg.grid_size
        for i in range(num_drones):
            if reached[i]:
                per_sorted.append([self.stay_action_index])
            else:
                # Base Manhattan distance
                base_dist = np.sum(np.abs(state[i, :3] + self.action_vectors - self.goals[i]), axis=1)

                # Convert penalties (negative values) to positive cost increases
                adj = base_dist.astype(float).copy()
                for a in range(self.num_per_drone_actions):
                    if a == self.stay_action_index:
                        tgt = state[i, :3]
                    else:
                        tgt = state[i, :3] + self.action_vectors[a]

                    # Obstacle penalty: closer to obstacle = higher cost
                    tx, ty, tz = np.clip(tgt, [0, 0, 0], [X-1, Y-1, Z-1]).astype(int)
                    obs_dist = float(self.env.dist_field[tx, ty, tz])
                    # Apply penalty when distance < 3, heavier penalty when closer
                    obstacle_penalty = max(0, 3.0 - obs_dist) * 3.0

                    # Symmetry breaking and swap risk penalties
                    sym_pen = self._right_of_way_penalty(i, state, tgt)   # <= 0
                    swap_pen = self._swap_risk_penalty(i, state, tgt)     # <= 0

                    # Combine all penalties (negative penalties become positive costs)
                    adj[a] += obstacle_penalty - (sym_pen + swap_pen)

                # Sort ascending (lower is better)
                per_sorted.append(list(np.argsort(adj)))

        # Conflict resolution: start from each drone's best, losers backoff
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
                    # Fallback: random action to maintain exploration
                    p_idx[i] = random.randint(0, len(per_sorted[i]) - 1)

        # Still conflicting: global random
        return random.randint(0, self.env.num_actions - 1)

    # ==================== Utilities and heuristics ====================
    def _active_drones_mask(self, state: np.ndarray):
        reached = state[:, 3].astype(bool)
        active_mask = ~reached
        return active_mask, reached

    def _is_endgame(self, state: np.ndarray) -> bool:
        active_mask, _ = self._active_drones_mask(state)
        return int(active_mask.sum()) <= self.end_active_threshold

    def _heuristic_value(self, state: np.ndarray) -> float:
        """
        Heuristic for rollout terminal value estimation.

        Components:
        1. Goal distance (primary) - negative Manhattan distance to goals
        2. Separation bonus - encourage spread to reduce congestion
        """
        active_mask, _ = self._active_drones_mask(state)
        if not active_mask.any():
            return 0.0

        pos = state[:, :3]
        d_goal = np.sum(np.abs(pos[active_mask] - self.goals[active_mask]), axis=1).sum()

        # Separation bonus (reduce congestion)
        P = pos[active_mask]
        sep = 0.0
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                sep += np.max(np.abs(P[i] - P[j]))

        return self.heuristic_alpha * float(d_goal) + 0.08 * float(sep)

    # Symmetry breaking and swap risk penalty (critical patch)
    def _swap_risk_penalty(self, i: int, state: np.ndarray, tgt: np.ndarray) -> float:
        """
        If planning to move to another drone's current cell while that drone's goal direction aligns
        with moving towards me, apply heavy penalty to avoid high-risk position swapping.
        """
        me = state[i, :3]
        pen = 0.0
        for j in range(state.shape[0]):
            if j == i or state[j, 3] == 1:
                continue
            other = state[j, :3]
            if np.array_equal(tgt, other):
                dir_to_me = np.sign(me - other)
                goal_dir = np.sign(self.goals[j] - other)
                if np.dot(dir_to_me[:2], goal_dir[:2]) > 0:  # Horizontal direction aligned -> likely coming towards me
                    pen -= 6.0
        return pen

    def _right_of_way_penalty(self, i: int, state: np.ndarray, tgt: np.ndarray) -> float:
        """
        Simple right-of-way rule: when approaching head-on in same row/column,
        lower-indexed drone has priority; higher-indexed drone gets penalty to encourage sidestepping/waiting.
        """
        me = state[i, :3]
        step = tgt - me
        if abs(step[2]) != 0 or (abs(step[0]) + abs(step[1]) != 1):
            return 0.0  # Only trigger for horizontal single-step moves
        pen = 0.0
        for j in range(state.shape[0]):
            if j == i or state[j, 3] == 1:
                continue
            other = state[j, :3]
            same_row = (other[1] == me[1]) and (step[1] == 0) and (np.sign(other[0]-me[0]) == np.sign(step[0]))
            same_col = (other[0] == me[0]) and (step[0] == 0) and (np.sign(other[1]-me[1]) == np.sign(step[1]))
            if same_row or same_col:
                if not (i < j):  # Lower index has priority
                    pen -= 2.5
        return pen

    # ==================== Adaptive tuning ====================
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
        self.c = float(np.clip(base_c * (1.0 + 0.5 * rho), 0.5, self.c_max))
        self.epsilon = max(0.05, min(0.05 + 0.15 * rho, 0.25))
        self.max_rollout_depth = int(max(15, min(round(0.8 * dbar) + (10 if rho > 0.5 else 0), 40)))
        self.topk_per_drone = max(4, min(4 + int(round(2 * rho)), 8))
        self.max_prior_samples = int(max(80, min(100 + 80 * rho + 0.1 * A, 220)))
        self.apw_alpha = max(0.5, min(0.5 + 0.2 * rho, 0.8))
        self.apw_k = 8

    def _online_adjust(self, root: "Node"):
        # Endgame detection: when some drones have reached, reduce exploration for precise final approach
        num_reached = int(np.sum(root.state[:, 3] == 1))
        num_total = root.state.shape[0]

        # If more than half of drones have reached, decay c to shift towards exploitation
        if num_reached >= num_total // 2:
            old_c = self.c
            self.c *= 0.85  # Revert to previous decay rate
            self.c = float(np.clip(self.c, 0.01, self.c_max))
            if np.isnan(self.c) or np.isinf(self.c):
                self.c = 1.0
            # Only print on first entry to endgame or if c changes > 10%, to avoid spam
            if not hasattr(self, '_in_endgame') or abs(old_c - self.c) / old_c > 0.1:
                print(f"[ENDGAME] {num_reached}/{num_total} reached | c: {old_c:.2f} -> {self.c:.2f}")
                self._in_endgame = True
            return  # Skip regular adjustment

        # Early phase: regular adjustment based on visit distribution
        totalN = sum(ch.visits for ch in root.children.values()) + 1e-9
        maxN = max((ch.visits for ch in root.children.values()), default=0)
        p_star = maxN / totalN
        old_c = self.c

        if p_star < 0.35:      # Too dispersed → increase exploration
            self.c += np.tanh(0.15)
            adj_type = "INC"
        elif p_star > 0.60:    # Too concentrated → decrease exploration
            self.c += -np.tanh(0.10)
            adj_type = "DEC"
        else:
            adj_type = "---"

        self.c = float(np.clip(self.c, 0.01, self.c_max))
        if np.isnan(self.c) or np.isinf(self.c):
            self.c = 3.0

        if adj_type != "---":
            print(f"[EARLY] p*={p_star:.2f} {adj_type} | c: {old_c:.2f} -> {self.c:.2f}")

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
# Optional CLI parameter interface
# ------------------------------
def add_planner_args(parser):
    p = parser
    # Basic parameters
    p.add_argument("--exploration_constant", type=float, default=3.0)
    p.add_argument("--rollout_epsilon", type=float, default=0.2)
    p.add_argument("--max_rollout_depth", type=int, default=20)
    # Prior/APW
    p.add_argument("--apw_k", type=float, default=10.0)
    p.add_argument("--apw_alpha", type=float, default=0.5)
    p.add_argument("--topk_per_drone", type=int, default=5)
    p.add_argument("--max_prior_samples", type=int, default=100)
    # Endgame (recommended 1 for better convergence; can be overridden)
    p.add_argument("--end_active_threshold", type=int, default=1)
    p.add_argument("--end_rollout_depth", type=int, default=60)
    p.add_argument("--end_epsilon", type=float, default=0.1)
    p.add_argument("--end_c_scale", type=float, default=0.4)
    p.add_argument("--heuristic_alpha", type=float, default=-5.0)
    p.add_argument("--stay_action_index", type=int, default=0)
    # Compatibility (allow CLI input even if not currently used, to avoid errors)
    p.add_argument("--use_clearance_gain", action="store_true", default=True)
    p.add_argument("--no_use_clearance_gain", dest="use_clearance_gain", action="store_false")
    p.add_argument("--use_lateral_bias", action="store_true", default=True)
    p.add_argument("--no_use_lateral_bias", dest="use_lateral_bias", action="store_false")
    p.add_argument("--asym_tau", type=float, default=0.12)
    # Adaptive parameters
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
        # CLI compatibility
        use_clearance_gain=args.use_clearance_gain,
        use_lateral_bias=args.use_lateral_bias,
        asym_tau=args.asym_tau,
        # Adaptive parameters
        auto_param=args.auto_param,
        auto_adjust_period=args.auto_adjust_period,
        adapt_log=args.adapt_log,
    )
