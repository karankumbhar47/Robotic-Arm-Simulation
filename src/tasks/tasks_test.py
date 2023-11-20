"""Integration tests for ravens tasks."""

from absl.testing import absltest
from absl.testing import parameterized
from ravens import tasks
from ravens.environments import environment

ASSETS_PATH = 'ravens/environments/assets/'

class TaskTest(parameterized.TestCase):

  def _create_env(self, continuous=False):
    """Create and configure the environment based on the task type.

    Args:
      continuous (bool): Whether to create a continuous environment.

    Returns:
      Environment: The configured environment instance.
    """
    assets_root = ASSETS_PATH
    if continuous:
      env = environment.ContinuousEnvironment(assets_root)
    else:
      env = environment.Environment(assets_root)
    env.seed(0)
    return env

  def _run_oracle_in_env(self, env, max_steps=10):
    """Run the oracle agent in the environment for a specified number of steps.

    Args:
      env (Environment): The environment instance.
      max_steps (int): The maximum number of steps to run the oracle agent.
    """
    agent = env.task.oracle(env)
    obs = env.reset()
    info = None
    done = False
    for _ in range(max_steps):
      act = agent.act(obs, info)
      obs, _, done, info = env.step(act)
      if done:
        break

  @parameterized.named_parameters(
      ('BlockInsertion', tasks.BlockInsertion()),
  )
  def test_all_tasks(self, ravens_task):
    """Run tests for tasks with discrete actions."""
    env = self._create_env()
    env.set_task(ravens_task)
    self._run_oracle_in_env(env)

  @parameterized.named_parameters(
      ('BlockInsertion', tasks.BlockInsertion(continuous=True)),
  )
  def test_all_tasks_continuous(self, ravens_task):
    """Run tests for tasks with continuous actions."""
    env = self._create_env(continuous=True)
    env.set_task(ravens_task)
    self._run_oracle_in_env(env, max_steps=200)

if __name__ == '__main__':
  absltest.main()
