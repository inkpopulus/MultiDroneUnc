# my_planner.py

import numpy as np
import math
import time
import random
from multi_drone import MultiDroneUnc


class Node:
    """MCTS搜索树的节点"""

    def __init__(self, state, parent=None, action=None):
        self.state = np.array(state, dtype=np.int32)
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None
        self.is_terminal = False
        self.depth = 0

    def q_value(self):
        """计算节点的平均回报 Q̄(s)"""
        if self.visits == 0:
            return 0.0
        return self.value / self.visits

    def is_fully_expanded(self):
        """检查是否所有动作都已尝试"""
        return len(self.untried_actions) == 0


class MyPlanner:
    def __init__(self, env: MultiDroneUnc, exploration_constant: float = 1.41, rollout_epsilon: float = 0.1):
        self.env = env
        self.c = exploration_constant
        self.epsilon = rollout_epsilon  # Epsilon for epsilon-greedy rollout
        self.discount_factor = env.get_config().discount_factor
        self.max_rollout_depth = env.get_config().max_num_steps

    def plan(self, current_state: np.ndarray, planning_time_per_step: float) -> int:
        # 检查动作空间是否为空
        if self.env.num_actions == 0:
            return 0

        # 用一个虚拟动作检查是否已终止
        _, _, root_done, _ = self.env.simulate(current_state, 0)
        if root_done:
            return 0

        root = Node(state=current_state)
        root.untried_actions = list(range(self.env.num_actions))
        random.shuffle(root.untried_actions)
        root.depth = 0

        start_time = time.time()
        while time.time() - start_time < planning_time_per_step:
            leaf_node = self._selection(root)

            if not leaf_node.is_terminal:
                child_node = self._expansion(leaf_node)
                rollout_reward = self._rollout(child_node.state, child_node.depth)
                self._backpropagation(child_node, rollout_reward)
            else:
                self._backpropagation(leaf_node, 0.0)

        if not root.children:
            return random.randint(0, self.env.num_actions - 1)

        best_action = max(root.children, key=lambda action: root.children[action].visits)
        return best_action

    def _selection(self, node: Node) -> Node:
        current_node = node
        while not current_node.is_terminal:
            if not current_node.is_fully_expanded():
                return current_node
            else:
                if not current_node.children:
                    return current_node
                current_node = self._best_child_uct(current_node)
        return current_node

    def _expansion(self, node: Node) -> Node:
        action = node.untried_actions.pop()
        next_state, reward, done, _ = self.env.simulate(node.state, action)

        child_node = Node(state=next_state, parent=node, action=action)
        child_node.is_terminal = done
        child_node.depth = node.depth + 1

        if not done:
            child_node.untried_actions = list(range(self.env.num_actions))
            random.shuffle(child_node.untried_actions)
        else:
            child_node.untried_actions = []

        node.children[action] = child_node
        # The reward from expansion is handled in backpropagation
        return child_node

    def _rollout(self, state: np.ndarray, depth: int) -> float:
        total_rollout_reward = 0.0
        current_state = state
        current_depth = depth
        discount = 1.0

        while current_depth < self.max_rollout_depth:
            action = self._rollout_policy(current_state)
            next_state, reward, done, _ = self.env.simulate(current_state, action)

            total_rollout_reward += discount * reward

            current_state = next_state
            discount *= self.discount_factor
            current_depth += 1
            if done:
                break
        return total_rollout_reward

    def _backpropagation(self, node: Node, rollout_reward: float):
        current_node = node
        current_return = rollout_reward

        while current_node is not None:
            current_node.visits += 1
            current_node.value += current_return

            # The reward from the step leading to this node needs to be added
            # before discounting for the parent.
            if current_node.parent is not None:
                _, reward_to_node, _, _ = self.env.simulate(current_node.parent.state, current_node.action)
                current_return = reward_to_node + self.discount_factor * current_return

            current_node = current_node.parent

    def _best_child_uct(self, node: Node) -> Node:
        log_parent_visits = math.log(node.visits + 1e-10)
        best_score = -float('inf')
        best_child = None

        for child in node.children.values():
            exploit_term = child.q_value()
            explore_term = self.c * math.sqrt(log_parent_visits / (child.visits + 1e-10))
            score = exploit_term + explore_term
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _rollout_policy(self, state: np.ndarray) -> int:
        """
        Epsilon-Greedy Rollout Policy with collision avoidance heuristic.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.env.num_actions - 1)

        num_drones = state.shape[0]
        action_vectors = self.env._action_vectors
        num_single_actions = action_vectors.shape[0]

        # Calculate sorted distances for each drone to its goal
        all_dists = []
        all_indices = []
        for i in range(num_drones):
            if state[i, 3]:  # Drone has reached goal
                dists = np.full(num_single_actions, float('inf'))
                indices = np.arange(num_single_actions)
            else:
                current_pos = state[i, :3]
                goal_pos = self.env.goals[i]
                potential_next_positions = current_pos + action_vectors
                distances = np.sum(np.abs(potential_next_positions - goal_pos), axis=1)
                indices = np.argsort(distances)
                dists = distances[indices]
            all_dists.append(dists)
            all_indices.append(indices)

        # Iteratively build a collision-free joint action
        chosen_actions = np.zeros(num_drones, dtype=np.int32)
        p_indices = np.zeros(num_drones, dtype=np.int32)

        resolved = False
        while not resolved:
            potential_targets = []
            for i in range(num_drones):
                action_idx = all_indices[i][p_indices[i]]
                chosen_actions[i] = action_idx
                target_pos = state[i, :3] + action_vectors[action_idx]
                potential_targets.append(tuple(target_pos))

            # Check for collisions
            unique_targets = set(potential_targets)
            if len(unique_targets) == num_drones:
                resolved = True
            else:
                # Find the drone with the worst alternative and make it yield
                worst_drone = -1
                max_diff = -float('inf')

                counts = {tgt: potential_targets.count(tgt) for tgt in unique_targets}
                colliding_targets = {tgt for tgt, count in counts.items() if count > 1}

                for i in range(num_drones):
                    if tuple(potential_targets[i]) in colliding_targets:
                        if p_indices[i] + 1 < num_single_actions:
                            current_dist = all_dists[i][p_indices[i]]
                            next_dist = all_dists[i][p_indices[i] + 1]
                            diff = next_dist - current_dist
                            if diff > max_diff:
                                max_diff = diff
                                worst_drone = i
                        else:  # No more alternatives, this drone is stuck
                            worst_drone = i
                            break

                if worst_drone != -1:
                    if p_indices[worst_drone] + 1 < num_single_actions:
                        p_indices[worst_drone] += 1
                    else:  # Can't resolve, break and use current (colliding) actions
                        break
                else:  # Should not happen if there are collisions
                    break

        return self.env._encode_action(chosen_actions)

