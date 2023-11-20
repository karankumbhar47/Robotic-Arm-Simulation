"""Insertion Tasks."""

import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils

import pybullet as p


class BlockInsertion(Task):
  """Insertion Task - Base Variant."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.max_steps = 3

  def reset(self, env):
    super().reset(env)
    block_id = self.add_block(env)
    targ_pose = self.add_fixture(env)
    self.goals.append(([(block_id, (2 * np.pi, None))], np.int32([[1]]),
                       [targ_pose], False, True, 'pose', None, 1))

  def add_block(self, env):
    """Adding L-shaped block."""
    size = (0.1, 0.1, 0.04)
    urdf = 'insertion/ell.urdf'
    pose = self.get_random_pose(env, size)
    return env.add_object(urdf, pose)

  def add_fixture(self, env):
    """Adding L-shaped fixture to place block."""
    size = (0.1, 0.1, 0.04)
    urdf = 'insertion/fixture.urdf'
    pose = self.get_random_pose(env, size)
    env.add_object(urdf, pose, 'fixed')
    return pose


class BlockInsertionTranslation(BlockInsertion):
  """Insertion Task - Translation Variant."""

  def get_random_pose(self, env, obj_size):
    pose = super(BlockInsertionTranslation, self).get_random_pose(env, obj_size)
    pos, rot = pose
    rot = utils.eulerXYZ_to_quatXYZW((0, 0, np.pi / 2))
    return pos, rot



class BlockInsertionEasy(BlockInsertionTranslation):
  """Insertion Task - Easy Variant."""

  def add_block(self, env):
    """Add L-shaped block in fixed position."""
    urdf = 'insertion/ell.urdf'
    pose = ((0.5, 0, 0.02), p.getQuaternionFromEuler((0, 0, np.pi / 2)))
    return env.add_object(urdf, pose)


class BlockInsertionSixDof(BlockInsertion):
  """Insertion Task - 6DOF Variant."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.sixdof = True
    self.pos_eps = 0.02

  def add_fixture(self, env):
    """Add L-shaped fixture to place block."""
    size = (0.1, 0.1, 0.04)
    urdf = 'insertion/fixture.urdf'
    pose = self.get_random_pose_6dof(env, size)
    env.add_object(urdf, pose, 'fixed')
    return pose

  def get_random_pose_6dof(self, env, obj_size):
    pos, rot = super(BlockInsertionSixDof, self).get_random_pose(env, obj_size)
    z = (np.random.rand() / 10) + 0.03
    pos = (pos[0], pos[1], obj_size[2] / 2 + z)
    roll = (np.random.rand() - 0.5) * np.pi / 2
    pitch = (np.random.rand() - 0.5) * np.pi / 2
    yaw = np.random.rand() * 2 * np.pi
    rot = utils.eulerXYZ_to_quatXYZW((roll, pitch, yaw))
    return pos, rot


class BlockInsertionNoFixture(BlockInsertion):
  """Insertion Task - No Fixture Variant."""

  def add_fixture(self, env):
    """Add target pose to place block."""
    size = (0.1, 0.1, 0.04)
    pose = self.get_random_pose(env, size)
    return pose
