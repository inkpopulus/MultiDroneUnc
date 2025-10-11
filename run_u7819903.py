import argparse
from multi_drone import MultiDroneUnc
from my_planner import MyPlanner

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True,
                    help="Path to the yaml configuration file")
parser.add_argument('--planning_time', type=float, default=1.0,
                    help="Planning time per step in seconds")
parser.add_argument('--exploration_constant', type=float, default=1.41,
                    help="UCB1 exploration constant")
parser.add_argument('--rollout_epsilon', type=float, default=0.1,
                    help="Epsilon for epsilon-greedy rollout policy")
args = parser.parse_args()

def run(env, planner, planning_time_per_step=1.0):
    """运行规划循环"""
    current_state = env.reset()
    num_steps = 0
    total_discounted_reward = 0.0
    history = []
    print(f"Starting planning loop with {planning_time_per_step:.2f}s per step.")

    while True:
        action = planner.plan(current_state, planning_time_per_step)
        next_state, reward, done, info = env.step(action)

        total_discounted_reward += (env.get_config().discount_factor ** num_steps) * reward
        history.append((current_state, action, reward, next_state, done, info))

        print(f"Step: {num_steps + 1}, Action: {action}, Reward: {reward:.2f}, Done: {done}")

        current_state = next_state
        num_steps += 1

        if done or num_steps >= env.get_config().max_num_steps:
            break

    return total_discounted_reward, history

# Instantiate the environment with the given config
env = MultiDroneUnc(args.config)

# Instantiate the planner
planner = MyPlanner(
    env,
    exploration_constant=args.exploration_constant,
    rollout_epsilon=args.rollout_epsilon
)

# Run the planning loop
total_discounted_reward, history = run(
    env,
    planner,
    planning_time_per_step=args.planning_time
)

print("\n" + "="*20 + " RESULTS " + "="*20)
print(f"Success: {history[-1][5]['success']}")
print(f"Total discounted reward: {total_discounted_reward:.2f}")
print(f"Number of steps: {len(history)}")
print("="*50 + "\n")


# Show visualization
env.show()