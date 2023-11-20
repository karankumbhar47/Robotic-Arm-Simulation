"""Classes to handle gripper dynamics."""

import os

import numpy as np
from ravens.utils import pybullet_utils

import pybullet as p

SPATULA_BASE_URDF = 'ur5/spatula/spatula-base.urdf'
SUCTION_BASE_URDF = 'ur5/suction/suction-base.urdf'
SUCTION_HEAD_URDF = 'ur5/suction/suction-head.urdf'


class Gripper:
    """Base gripper class."""

    def __init__(self, assets_root):
        """
        Initialize the gripper.

        Parameters:
            assets_root (str): Root directory with assets.
        """
        self.assets_root = assets_root
        self.activated = False

    def step(self):
        """This function can be used to create gripper-specific behaviors."""
        return

    def activate(self, objects):
        """
        Activate the gripper.

        Parameters:
            objects: Objects to interact with.
        """
        del objects
        return

    def release(self):
        """Release any gripped object."""
        return


class Spatula(Gripper):
    """Simulate a simple spatula for pushing."""

    def __init__(self, assets_root, robot, ee, obj_ids):
        """
        Create spatula and 'attach' it to the robot.

        Parameters:
            assets_root (str): Root directory with assets.
            robot (int): PyBullet ID of the robot.
            ee (int): PyBullet ID of the end effector link.
            obj_ids: List of PyBullet IDs of all objects in the environment.
        """
        super().__init__(assets_root)

        # Load spatula model.
        pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
        base = pybullet_utils.load_urdf(
            p, os.path.join(self.assets_root, SPATULA_BASE_URDF), pose[0], pose[1])
        p.createConstraint(
            parentBodyUniqueId=robot,
            parentLinkIndex=ee,
            childBodyUniqueId=base,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, 0.01))


class Suction(Gripper):
    """Simulate simple suction dynamics."""

    def __init__(self, assets_root, robot, ee, obj_ids):
        """
        Create suction and 'attach' it to the robot.

        Parameters:
            assets_root (str): Root directory with assets.
            robot (int): PyBullet ID of the robot.
            ee (int): PyBullet ID of the end effector link.
            obj_ids: List of PyBullet IDs of all suctionable objects in the environment.
        """
        super().__init__(assets_root)

        # Load suction gripper base model (visual only).
        pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
        base = pybullet_utils.load_urdf(
            p, os.path.join(self.assets_root, SUCTION_BASE_URDF), pose[0], pose[1])
        p.createConstraint(
            parentBodyUniqueId=robot,
            parentLinkIndex=ee,
            childBodyUniqueId=base,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, 0.01))

        # Load suction tip model (visual and collision) with compliance.
        pose = ((0.487, 0.109, 0.347), p.getQuaternionFromEuler((np.pi, 0, 0)))
        self.body = pybullet_utils.load_urdf(
            p, os.path.join(self.assets_root, SUCTION_HEAD_URDF), pose[0], pose[1])
        constraint_id = p.createConstraint(
            parentBodyUniqueId=robot,
            parentLinkIndex=ee,
            childBodyUniqueId=self.body,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.08))
        p.changeConstraint(constraint_id, maxForce=50)

        # Reference to object IDs in the environment for simulating suction.
        self.obj_ids = obj_ids

        # Indicates whether gripper is gripping anything (rigid or def).
        self.activated = False

        # For gripping and releasing rigid objects.
        self.contact_constraint = None

        # Defaults for deformable parameters, and can override in tasks.
        self.def_ignore = 0.035  # TODO: check if this is needed
        self.def_threshold = 0.030
        self.def_nb_anchors = 1

        # Track which deformable is being gripped (if any), and anchors.
        self.def_grip_item = None
        self.def_grip_anchors = []

        # Determines release when gripped deformable touches a rigid/def.
        # TODO: should check if the code uses this -- not sure?
        self.def_min_vetex = None
        self.def_min_distance = None

        # Determines release when a gripped rigid touches defs (e.g. cloth-cover).
        self.init_grip_distance = None
        self.init_grip_item = None

    def activate(self):
        """Simulate suction using a rigid fixed constraint to contacted object."""
        # TODO: check deformables logic.
        # del def_ids

        if not self.activated:
            points = p.getContactPoints(bodyA=self.body, linkIndexA=0)
            if points:
                # Handle contact between suction with a rigid object.
                for point in points:
                    obj_id, contact_link = point[2], point[4]
                    if obj_id in self.obj_ids['rigid']:
                        body_pose = p.getLinkState(self.body, 0)
                        obj_pose = p.getBasePositionAndOrientation(obj_id)
                        world_to_body = p.invertTransform(body_pose[0], body_pose[1])
                        obj_to_body = p.multiplyTransforms(world_to_body[0],
                                                           world_to_body[1],
                                                           obj_pose[0], obj_pose[1])
                        self.contact_constraint = p.createConstraint(
                            parentBodyUniqueId=self.body,
                            parentLinkIndex=0,
                            childBodyUniqueId=obj_id,
                            childLinkIndex=contact_link,
                            jointType=p.JOINT_FIXED,
                            jointAxis=(0, 0, 0),
                            parentFramePosition=obj_to_body[0],
                            parentFrameOrientation=obj_to_body[1],
                            childFramePosition=(0, 0, 0),
                            childFrameOrientation=(0, 0, 0))

                self.activated = True

    def release(self):
        """Release gripper object, only applied if gripper is 'activated'."""
        if self.activated:
            self.activated = False

            # Release gripped rigid object (if any).
            if self.contact_constraint is not None:
                try:
                    p.removeConstraint(self.contact_constraint)
                    self.contact_constraint = None
                except:
                    pass
                self.init_grip_distance = None
                self.init_grip_item = None

            # Release gripped deformable object (if any).
            if self.def_grip_anchors:
                for anchor_id in self.def_grip_anchors:
                    p.removeConstraint(anchor_id)
                self.def_grip_anchors = []
                self.def_grip_item = None
                self.def_min_vetex = None
                self.def_min_distance = None

    def detect_contact(self):
        """Detects contact with a rigid object."""
        body, link = self.body, 0
        if self.activated and self.contact_constraint is not None:
            try:
                info = p.getConstraintInfo(self.contact_constraint)
                body, link = info[2], info[3]
            except:
                self.contact_constraint = None
                pass

        # Get all contact points between the suction and a rigid body.
        points = p.getContactPoints(bodyA=body, linkIndexA=link)
        if self.activated:
            points = [point for point in points if point[2] != self.body]

        # Check if contact is made with some rigid item.
        if points:
            return True

        return False

    def check_grasp(self):
        """Check a grasp (object in contact?) for picking success."""
        suctioned_object = None
        if self.contact_constraint is not None:
            suctioned_object = p.getConstraintInfo(self.contact_constraint)[2]
        return suctioned_object is not None
