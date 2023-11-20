"""Data collection script."""

import os

from absl import app
from absl import flags

import numpy as np

from src import tasks
from src.dataset import Dataset
from src.environments.environment import ContinuousEnvironment
from src.environments.environment import Environment

flags.DEFINE_string('assets_root', '.', '')
flags.DEFINE_string('data_dir', './dataset', '')
flags.DEFINE_bool('disp', False, '')
flags.DEFINE_bool('shared_memory', False, '')
flags.DEFINE_string('task', 'block-insertion', '')
flags.DEFINE_string('mode', 'train', '')
flags.DEFINE_integer('n', 1000, '')
flags.DEFINE_bool('continuous', False, '')
flags.DEFINE_integer('steps_per_seg', 3, '')

FLAGS = flags.FLAGS


def main(unused_argv):

    # Initialize environment and task.
    env_cls = ContinousEnvironment if FLAGS.continuous else Environment
    env = env_cls(
        FLAGS.assets_root,
        disp=FLAGS.disp,
        shared_memory=FLAGS.shared_memory,
        hz=480)
    task = tasks.names[FLAGS.task](continuous=FLAGS.continuous)
    task.mode = FLAGS.mode

    # Initialize scripted oracle agent and dataset.
    agent = task.oracle(env, steps_per_seg=FLAGS.steps_per_seg)
    dataset = Dataset(os.path.join(
        FLAGS.data_dir, f'{FLAGS.task}-{task.mode}'))

    # Train seeds are even and test seeds are odd.
    seed = dataset.max_seed
    if seed < 0:
        seed = -1 if (task.mode == 'test') else -2

    # Determine max steps per episode.
    max_steps = task.max_steps
    if FLAGS.continuous:
        max_steps *= (FLAGS.steps_per_seg * agent.num_poses)

    # Collect training data from oracle demonstrations.
    while dataset.n_episodes < FLAGS.n:
        print(f'Oracle demonstration: {dataset.n_episodes + 1}/{FLAGS.n}')
        episode, total_reward = [], 0
        seed += 2
        np.random.seed(seed)
        env.set_task(task)
        obs = env.reset()
        info = None
        reward = 0
        for _ in range(max_steps):
            act = agent.act(obs, info)
            episode.append((obs, act, reward, info))
            obs, reward, done, info = env.step(act)
            print(obs,reward,done,info)
            total_reward += reward

            print(f'Total Reward: {total_reward} Done: {done}')
            if done:
                break
        episode.append((obs, None, reward, info))

        # Only save completed demonstrations.
        # TODO(andyzeng): add back deformable logic.
        if total_reward > 0.99:
            dataset.add(seed, episode)


if __name__ == '__main__':
    app.run(main)
