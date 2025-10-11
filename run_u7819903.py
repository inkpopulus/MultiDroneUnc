import argparse
import time
from multi_drone import MultiDroneUnc
# 确保您使用的是最新的、修复了bug的my_planner
from my_planner import MyPlanner

parser = argparse.ArgumentParser(description="Run MCTS planner for Multi-Drone environment.")
parser.add_argument('--config', type=str, required=True,
                    help="Path to the yaml configuration file")
parser.add_argument('--planning_time', type=float,
                    help="Planning time per step in seconds. Overrides --num_iterations if both are set.")
parser.add_argument('--num_iterations', type=int,
                    help="Number of MCTS iterations per step. Used if --planning_time is not set.")
parser.add_argument('--exploration_constant', type=float, default=30.0,
                    help="UCB1 exploration constant")
parser.add_argument('--rollout_epsilon', type=float, default=0.1,
                    help="Epsilon for epsilon-greedy rollout policy")
parser.add_argument('--max_rollout_depth', type=int,
                    help="Maximum depth for each rollout simulation.")
args = parser.parse_args()

# 设置默认规划方法
if args.planning_time is None and args.num_iterations is None:
    args.planning_time = 1.0  # 默认使用1秒时间


def run(env, planner):
    """运行规划循环"""
    current_state = env.reset()
    num_steps = 0
    total_discounted_reward = 0.0
    history = []
    total_planning_time = 0.0  # <--- 新增：总规划时间计时器

    planning_method = "time" if args.planning_time is not None else "iterations"
    planning_limit = args.planning_time if planning_method == "time" else args.num_iterations
    print(f"Starting planning loop with limit per step: {planning_limit} {planning_method}")

    while True:
        # --- 计时开始 ---
        step_start_time = time.time()

        action = planner.plan(current_state,
                              planning_time_per_step=args.planning_time,
                              num_iterations=args.num_iterations)

        # --- 计时结束 ---
        step_planning_time = time.time() - step_start_time
        total_planning_time += step_planning_time

        next_state, reward, done, info = env.step(action)

        total_discounted_reward += (env.get_config().discount_factor ** num_steps) * reward
        history.append((current_state, action, reward, next_state, done, info))

        print(
            f"Step: {num_steps + 1}, Action: {action}, Reward: {reward:.2f}, Done: {done}, Planning Time: {step_planning_time:.2f}s")

        current_state = next_state
        num_steps += 1

        if done or num_steps >= env.get_config().max_num_steps:
            break

    return total_discounted_reward, history, total_planning_time


# 实例化环境
env = MultiDroneUnc(args.config)

# 实例化规划器
planner = MyPlanner(
    env,
    exploration_constant=args.exploration_constant,
    rollout_epsilon=args.rollout_epsilon,
    max_rollout_depth=args.max_rollout_depth
)

# 运行规划循环
total_discounted_reward, history, total_planning_time = run(env, planner)

# 打印最终结果
print("\n" + "=" * 20 + " RESULTS " + "=" * 20)
print(f"Success: {history[-1][5]['success']}")
print(f"Total discounted reward: {total_discounted_reward:.2f}")
print(f"Number of steps: {len(history)}")
print(f"Total planning time: {total_planning_time:.2f}s")  # <--- 新增：打印总时间
print(f"Average planning time per step: {total_planning_time / len(history):.2f}s")
print("=" * 50 + "\n")

# 显示可视化结果
env.show()