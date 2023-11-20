"""Insertion Tasks."""

import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils

import pybullet as p


class BlockInsertion(Task):
    """Base Variant of the Insertion Task."""

    def __init__(self, *args, **kwargs):
        """
        Initialize the BlockInsertion task.

        Parameters:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.max_steps = 3

    def reset(self, env):
        """
        Reset the environment for the BlockInsertion task.

        Parameters:
            env: The environment in which the task is being performed.
        """
        super().reset(env)
        block_id = self.add_block(env)
        targ_pose = self.add_fixture(env)
        self.goals.append(([(block_id, (2 * np.pi, None))], np.int32([[1]]),
                           [targ_pose], False, True, 'pose', None, 1))

    def add_block(self, env):
        """
        Add an L-shaped block to the environment.

        Parameters:
            env: The environment in which the block is added.

        Returns:
            block_id: The identifier for the added block.
        """
        size = (0.1, 0.1, 0.04)
        urdf = 'insertion/ell.urdf'
        pose = self.get_random_pose(env, size)
        return env.add_object(urdf, pose)

    def add_fixture(self, env):
        """
        Add an L-shaped fixture to place the block.

        Parameters:
            env: The environment in which the fixture is added.

        Returns:
            pose: The pose of the added fixture.
        """
        size = (0.1, 0.1, 0.04)
        urdf = 'insertion/fixture.urdf'
        pose = self.get_random_pose(env, size)
        env.add_object(urdf, pose, 'fixed')
        return pose


class BlockInsertionTranslation(BlockInsertion):
    """Translation Variant of the Insertion Task."""

    def get_random_pose(self, env, obj_size):
        """
        Get a random pose for the object.

        Parameters:
            env: The environment in which the pose is generated.
            obj_size: The size of the object.

        Returns:
            pose: The randomly generated pose.
        """
        pose = super(BlockInsertionTranslation, self).get_random_pose(env, obj_size)
        pos, rot = pose
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, np.pi / 2))
        return pos, rot


class BlockInsertionEasy(BlockInsertionTranslation):
    """Easy Variant of the Insertion Task."""

    def add_block(self, env):
        """
        Add an L-shaped block in a fixed position.

        Parameters:
            env: The environment in which the block is added.

        Returns:
            block_id: The identifier for the added block.
        """
        urdf = 'insertion/ell.urdf'
        pose = ((0.5, 0, 0.02), p.getQuaternionFromEuler((0, 0, np.pi / 2)))
        return env.add_object(urdf, pose)


class BlockInsertionSixDof(BlockInsertion):
    """6DOF Variant of the Insertion Task."""

    def __init__(self, *args, **kwargs):
        """
        Initialize the BlockInsertionSixDof task.

        Parameters:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.sixdof = True
        self.pos_eps = 0.02

    def add_fixture(self, env):
        """
        Add an L-shaped fixture with 6 degrees of freedom to place the block.

        Parameters:
            env: The environment in which the fixture is added.

        Returns:
            pose: The pose of the added fixture.
        """
        size = (0.1, 0.1, 0.04)
        urdf = 'insertion/fixture.urdf'
        pose = self.get_random_pose_6dof(env, size)
        env.add_object(urdf, pose, 'fixed')
        return pose

    def get_random_pose_6dof(self, env, obj_size):
        """
        Get a random pose with 6 degrees of freedom for the object.

        Parameters:
            env: The environment in which the pose is generated.
            obj_size: The size of the object.

        Returns:
            pose: The randomly generated pose.
        """
        pos, rot = super(BlockInsertionSixDof, self).get_random_pose(env, obj_size)
        z = (np.random.rand() / 10) + 0.03
        pos = (pos[0], pos[1], obj_size[2] / 2 + z)
        roll = (np.random.rand() - 0.5) * np.pi / 2
        pitch = (np.random.rand() - 0.5) * np.pi / 2
        yaw = np.random.rand() * 2 * np.pi
        rot = utils.eulerXYZ_to_quatXYZW((roll, pitch, yaw))
        return pos, rot


class BlockInsertionNoFixture(BlockInsertion):
    """No Fixture Variant of the Insertion Task."""

    def add_fixture(self, env):
        """
        Add a target pose to place the block.

        Parameters:
            env: The environment in which the target pose is added.

        Returns:
            pose: The target pose for placing the block.
        """
        size = (0.1, 0.1, 0.04)
        pose = self.get_random_pose(env, size)
        return pose
