import numpy as np
import math
import time
import random
from collections import defaultdict
from multi_drone import MultiDroneUnc


class Node:
    """MCTS搜索树的节点"""

    def __init__(self, state, parent=None, action=None, reward=0.0):
        self.state = np.array(state, dtype=np.int32)
        self.parent = parent
        self.action = action
        self.reward_from_parent = reward
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None
        self.is_terminal = False
        self.depth = parent.depth + 1 if parent is not None else 0

    def q_value(self):
        if self.visits == 0: return 0.0
        return self.value / self.visits

    def is_fully_expanded(self):
        return not self.untried_actions


class MyPlanner:
    """
    APW-MCTS规划器 (优化版)
    - 使用APW处理大动作空间
    - 不使用Tree Reuse (在此问题上效果不好)
    - 使用稳健的最终动作选择策略
    """

    def __init__(self, env: MultiDroneUnc,
                 exploration_constant: float = None,
                 rollout_epsilon: float = 0.1,
                 max_rollout_depth: int = None,
                 apw_k: float = 10.0,
                 apw_alpha: float = 0.5):
        self.env = env
        self.epsilon = rollout_epsilon
        self.discount_factor = env.get_config().discount_factor
        self.apw_k = apw_k
        self.apw_alpha = apw_alpha

        if exploration_constant is None:
            num_actions = env.num_actions
            self.c = 50.0 if num_actions > 1000 else 10.0
        else:
            self.c = exploration_constant

        self.max_rollout_depth = min(20, env.get_config().max_num_steps) if max_rollout_depth is None else max_rollout_depth

        self.goals = np.array(env.get_config().goal_positions)
        self.grid_size = env.get_config().grid_size
        self.action_vectors = self.env._action_vectors
        self.num_per_drone_actions = self.action_vectors.shape[0]

    def plan(self, current_state: np.ndarray,
             planning_time_per_step: float = None,
             num_iterations: int = None) -> int:

        _, _, root_done, _ = self.env.simulate(current_state, 0)
        if root_done: return 0

        # 不使用Tree Reuse - 每步都创建新树
        root = Node(state=current_state)
        root.untried_actions = self._get_prioritized_actions(current_state)

        use_time_limit = planning_time_per_step is not None
        start_time = time.time()
        iteration_count = 0

        while True:
            if use_time_limit:
                if time.time() - start_time > planning_time_per_step: break
            else:
                if num_iterations and iteration_count >= num_iterations: break

            self._run_mcts_iteration(root)
            iteration_count += 1

        if not root.children:
            return self._get_greedy_action(current_state)

        # 使用稳健的选择策略: 结合Q值和访问次数
        best_action = self._select_best_action_robust(root)
        return best_action

    def _run_mcts_iteration(self, root: Node):
        leaf_node = self._select_and_expand(root)
        rollout_reward = self._rollout(leaf_node.state, leaf_node.depth)
        self._backpropagate(leaf_node, rollout_reward)

    def _select_and_expand(self, node: Node) -> Node:
        current_node = node
        while not current_node.is_terminal:
            # APW核心逻辑
            if len(current_node.children) < self.apw_k * (current_node.visits ** self.apw_alpha):
                if not current_node.is_fully_expanded():
                    action = current_node.untried_actions.pop(0)
                    next_state, reward, done, _ = self.env.simulate(current_node.state, action)

                    child_node = Node(state=next_state, parent=current_node, action=action, reward=reward)
                    child_node.is_terminal = done

                    if not done:
                        child_node.untried_actions = self._get_prioritized_actions(next_state)

                    current_node.children[action] = child_node
                    return child_node

            if not current_node.children:
                return current_node

            current_node = self._best_child_uct(current_node)
        return current_node

    def _rollout(self, state: np.ndarray, depth: int) -> float:
        total_rollout_reward = 0.0
        current_state = np.copy(state)

        for i in range(self.max_rollout_depth):
            current_depth = depth + i
            if current_depth >= self.env.get_config().max_num_steps: break

            action = self._rollout_policy(current_state)
            next_state, reward, done, _ = self.env.simulate(current_state, action)

            total_rollout_reward += (self.discount_factor ** i) * reward

            current_state = next_state
            if done: break
        return total_rollout_reward

    def _backpropagate(self, node: Node, rollout_reward: float):
        current_node = node
        current_return = rollout_reward

        while current_node is not None:
            current_node.visits += 1
            current_node.value += current_return
            current_return = current_node.reward_from_parent + self.discount_factor * current_return
            current_node = current_node.parent

    def _best_child_uct(self, node: Node) -> Node:
        log_parent_visits = math.log(node.visits + 1)
        best_score = -float('inf')
        best_child = None

        for child in node.children.values():
            exploit_term = child.q_value()
            explore_term = self.c * math.sqrt(log_parent_visits / (child.visits + 1e-6))
            score = exploit_term + explore_term
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _select_best_action_robust(self, root: Node) -> int:
        """
        稳健的最终动作选择:
        1. 过滤掉访问次数太少的动作 (< 5次)
        2. 在剩余动作中选择Q值最高的
        3. 如果所有动作访问次数都少,选择访问次数最多的
        """
        if not root.children:
            return self._get_greedy_action(root.state)

        min_visits_threshold = 5  # 最小访问次数阈值

        # 候选动作: 访问次数 >= 阈值
        candidates = {
            action: child for action, child in root.children.items()
            if child.visits >= min_visits_threshold
        }

        if candidates:
            # 在候选中选择Q值最高的
            best_action = max(candidates, key=lambda a: candidates[a].q_value())
        else:
            # 所有动作访问次数都少,选择访问次数最多的
            best_action = max(root.children, key=lambda a: root.children[a].visits)

        return best_action

    def _get_greedy_action(self, state: np.ndarray) -> int:
        return self._rollout_policy(state, greedy=True)

    def _get_prioritized_actions(self, state: np.ndarray) -> list:
        num_drones = state.shape[0]
        positions = state[:, :3]
        reached = state[:, 3].astype(bool)

        per_drone_top = []
        top_k = min(5, self.num_per_drone_actions)

        for i in range(num_drones):
            if reached[i]:
                per_drone_top.append(list(range(self.num_per_drone_actions)))
            else:
                distances = np.sum(np.abs(
                    positions[i] + self.action_vectors - self.goals[i]
                ), axis=1)
                sorted_indices = np.argsort(distances)
                per_drone_top.append(sorted_indices[:top_k].tolist())

        prioritized = []
        best_combo = [top[0] for top in per_drone_top]
        prioritized.append(self.env._encode_action(np.array(best_combo, dtype=np.int32)))

        sample_size = min(100, self.env.num_actions)
        attempts = 0
        while len(prioritized) < sample_size and attempts < sample_size * 2:
            combo = [random.choice(top) for top in per_drone_top]
            action = self.env._encode_action(np.array(combo, dtype=np.int32))
            if action not in prioritized:
                prioritized.append(action)
            attempts += 1

        return prioritized

    def _rollout_policy(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.env.num_actions - 1)

        num_drones = state.shape[0]

        sorted_action_indices = []
        for i in range(num_drones):
            if state[i, 3]:
                sorted_action_indices.append(None)
            else:
                distances = np.sum(np.abs(state[i, :3] + self.action_vectors - self.goals[i]), axis=1)
                sorted_action_indices.append(np.argsort(distances))

        p_indices = np.zeros(num_drones, dtype=int)

        for _ in range(num_drones + 1):
            intended_actions = np.zeros(num_drones, dtype=int)
            targets = defaultdict(list)

            for i in range(num_drones):
                if sorted_action_indices[i] is None:
                    action_idx, target_pos = -1, tuple(state[i, :3])
                else:
                    action_idx = sorted_action_indices[i][p_indices[i]]
                    target_pos = tuple(state[i, :3] + self.action_vectors[action_idx])

                intended_actions[i] = action_idx
                targets[target_pos].append(i)

            colliding_drones = {drone for drones in targets.values() if len(drones) > 1 for drone in drones if
                                drone != min(drones)}

            if not colliding_drones:
                final_actions = np.array([a if a != -1 else 0 for a in intended_actions], dtype=int)
                return self.env._encode_action(final_actions)

            for i in colliding_drones:
                if sorted_action_indices[i] is not None:
                    if p_indices[i] + 1 < self.num_per_drone_actions:
                        p_indices[i] += 1
                    else:
                        p_indices[i] = random.randint(0, self.num_per_drone_actions - 1)

        return random.randint(0, self.env.num_actions - 1)
