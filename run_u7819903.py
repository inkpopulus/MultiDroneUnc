# run_planner.py
import argparse
import time
import sys
from multi_drone import MultiDroneUnc
from my_planner import MyPlanner, add_planner_args, make_planner_from_args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MCTS planner for Multi-Drone environment."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML configuration file (e.g., case2.yaml)"
    )
    parser.add_argument(
        "--planning_time", type=float,
        help="Planning time per step in seconds. If set, overrides --num_iterations."
    )
    parser.add_argument(
        "--num_iterations", type=int,
        help="Number of MCTS iterations per step. Used only if --planning_time is NOT set."
    )

    # 将所有 planner 参数注册进来（来自 my_planner.py）
    add_planner_args(parser)

    args = parser.parse_args()

    # 默认值：若两者都没给，默认按 1.0s/步
    if args.planning_time is None and args.num_iterations is None:
        args.planning_time = 1.0

    return args


def run(env: MultiDroneUnc, planner: MyPlanner, planning_time: float | None, num_iterations: int | None):
    """主循环：按时间/迭代限额逐步规划与执行"""
    current_state = env.reset()
    num_steps = 0
    total_discounted_reward = 0.0
    history = []  # (s, a, r, s', done, info)
    total_planning_time = 0.0

    planning_method = "time" if planning_time is not None else "iterations"
    planning_limit = planning_time if planning_method == "time" else num_iterations
    print(f"Starting planning loop with per-step limit: {planning_limit} {planning_method}")

    gamma = env.get_config().discount_factor
    max_steps = env.get_config().max_num_steps

    while True:
        step_start_time = time.time()

        action = planner.plan(
            current_state,
            planning_time_per_step=planning_time,
            num_iterations=num_iterations
        )

        step_planning_time = time.time() - step_start_time
        total_planning_time += step_planning_time

        next_state, reward, done, info = env.step(action)

        total_discounted_reward += (gamma ** num_steps) * reward
        history.append((current_state, action, reward, next_state, done, info))

        print(
            f"Step: {num_steps + 1:03d} | Action: {action} | "
            f"Reward: {reward:.2f} | Done: {done} | "
            f"PlanTime: {step_planning_time:.2f}s"
        )

        current_state = next_state
        num_steps += 1

        if done or num_steps >= max_steps:
            break

    return total_discounted_reward, history, total_planning_time


def main():
    args = parse_args()

    # 1) 实例化环境
    try:
        env = MultiDroneUnc(args.config)
    except Exception as e:
        print(f"[ERROR] Failed to create environment from config '{args.config}': {e}")
        sys.exit(1)

    # 2) 由命令行参数创建规划器（my_planner.py 提供）
    try:
        planner = make_planner_from_args(env, args)
    except Exception as e:
        print(f"[ERROR] Failed to create planner: {e}")
        sys.exit(1)

    # 3) 运行主循环
    total_discounted_reward, history, total_planning_time = run(
        env, planner, args.planning_time, args.num_iterations
    )

    # 4) 打印结果
    print("\n" + "=" * 20 + " RESULTS " + "=" * 20)
    success = None
    if len(history) > 0 and isinstance(history[-1][-1], dict) and "success" in history[-1][-1]:
        success = history[-1][-1]["success"]
    print(f"Success: {success}")
    print(f"Total discounted reward: {total_discounted_reward:.2f}")
    print(f"Number of steps: {len(history)}")
    print(f"Total planning time: {total_planning_time:.2f}s")
    if len(history) > 0:
        print(f"Average planning time per step: {total_planning_time / len(history):.2f}s")
    print("=" * 50 + "\n")

    # 5) 可视化（若环境支持）
    try:
        env.show()
    except Exception as e:
        print(f"[WARN] env.show() failed or not supported: {e}")


if __name__ == "__main__":
    main()
