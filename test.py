import gym
import torch.optim as optim

from model import VIN
from dqn_test import OptimizerSpec, dqn_testing
from utils.gym import get_env, get_wrapper_by_name
from utils.schedule import LinearSchedule
import argparse

BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 50000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 1
TARGER_UPDATE_FREQ = 10000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01

def main(env, num_timesteps, config):

    def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    dqn_testing(
        config=config,
        env=env,
        q_func=VIN,
        exploration=exploration_schedule,
    )

if __name__ == '__main__':
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lr',
        type=float,
        default=0.005,
        help='Learning rate, [0.01, 0.005, 0.002, 0.001]')
    parser.add_argument(
        '--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument(
        '--k', type=int, default=10, help='Number of Value Iterations')
    parser.add_argument(
        '--l_i', type=int, default=3, help='Number of channels in input layer')
    parser.add_argument(
        '--l_h',
        type=int,
        default=150,
        help='Number of channels in first hidden layer')
    parser.add_argument(
        '--l_q',
        type=int,
        default=9,
        help='Number of channels in q layer (~actions) in VI-module')
    parser.add_argument(
        '--batch_size', type=int, default=128, help='Batch size')
    config = parser.parse_args()

    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(seed)

    main(env, 1000000, config=config)
