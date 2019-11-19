"""
OpenPose is a tool that predicts human body joints from RGB images.
In this module, I define cross-project code that I need for reading outputs of openpose and drawing its body skeleton.
"""

import json
from more_itertools import chunked
from typing import List, Any, Dict, NamedTuple
from enum import Enum
import matplotlib.pyplot as plt


class OpenPoseJoints(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    MidHip = 8
    RHip = 9
    RKnee = 10
    RAnkle = 11
    LHip = 12
    LKnee = 13
    LAnkle = 14
    REye = 15
    LEye = 16
    REar = 17
    LEar = 18
    LBigToe = 19
    LSmallToe = 20
    LHeel = 21
    RBigToe = 22
    RSmallToe = 23
    RHeel = 24
    Background = 25


class JointDescriptor(NamedTuple):
    x: float
    y: float
    confidence: float
    joint: OpenPoseJoints


class OpenPoseBones(Enum):
    Nose_Neck = (OpenPoseJoints.Nose, OpenPoseJoints.Neck)
    Neck_RightShoulder = (OpenPoseJoints.Neck, OpenPoseJoints.RShoulder)
    RightShoulder_RightElbow = (OpenPoseJoints.RShoulder, OpenPoseJoints.RElbow)
    RightElbow_RightWrist = (OpenPoseJoints.RElbow, OpenPoseJoints.RWrist)
    Neck_LeftShoulder = (OpenPoseJoints.Neck, OpenPoseJoints.LShoulder)
    LeftShoulder_LeftElbow = (OpenPoseJoints.LShoulder, OpenPoseJoints.LElbow)
    LeftElbow_LeftWrist = (OpenPoseJoints.LElbow, OpenPoseJoints.LWrist)
    Neck_MidHip = (OpenPoseJoints.Neck, OpenPoseJoints.MidHip)
    MidHip_RightHip = (OpenPoseJoints.MidHip, OpenPoseJoints.RHip)
    RightHip_RightKnee = (OpenPoseJoints.RHip, OpenPoseJoints.RKnee)
    RightKnee_RightAnkle = (OpenPoseJoints.RKnee, OpenPoseJoints.RAnkle)
    MidHip_LeftHip = (OpenPoseJoints.MidHip, OpenPoseJoints.LHip)
    LeftHip_LeftKnee = (OpenPoseJoints.LHip, OpenPoseJoints.LKnee)
    LeftKnee_LeftAnkle = (OpenPoseJoints.LKnee, OpenPoseJoints.LAnkle)
    Nose_RightEye = (OpenPoseJoints.Nose, OpenPoseJoints.REye)
    Nose_LeftEye = (OpenPoseJoints.Nose, OpenPoseJoints.LEye)
    RightEye_RightEar = (OpenPoseJoints.REye, OpenPoseJoints.REar)
    LeftEye_LeftEar = (OpenPoseJoints.LEye, OpenPoseJoints.LEar)


JointsOfInterest: List[OpenPoseJoints] = [
    OpenPoseJoints.Nose,
    OpenPoseJoints.Neck,
    OpenPoseJoints.RShoulder,
    OpenPoseJoints.RElbow,
    OpenPoseJoints.RWrist,
    OpenPoseJoints.LShoulder,
    OpenPoseJoints.LElbow,
    OpenPoseJoints.LWrist,
    OpenPoseJoints.MidHip,
    OpenPoseJoints.RHip,
    OpenPoseJoints.RKnee,
    OpenPoseJoints.RAnkle,
    OpenPoseJoints.LHip,
    OpenPoseJoints.LKnee,
    OpenPoseJoints.LAnkle,
]

BonesOfInterest: List[OpenPoseBones] = [
    OpenPoseBones.Nose_Neck,
    OpenPoseBones.Neck_RightShoulder,
    OpenPoseBones.RightShoulder_RightElbow,
    OpenPoseBones.RightElbow_RightWrist,
    OpenPoseBones.Neck_LeftShoulder,
    OpenPoseBones.LeftShoulder_LeftElbow,
    OpenPoseBones.LeftElbow_LeftWrist,
    OpenPoseBones.Neck_MidHip,
    OpenPoseBones.MidHip_RightHip,
    OpenPoseBones.RightHip_RightKnee,
    OpenPoseBones.RightKnee_RightAnkle,
    OpenPoseBones.MidHip_LeftHip,
    OpenPoseBones.LeftHip_LeftKnee,
    OpenPoseBones.LeftKnee_LeftAnkle,
]


def read_openpose_json(filename: str) -> List[JointDescriptor]:
    with open(filename, "rb") as f:
        keypoints_list = []
        keypoints = json.load(f)
        assert (
            len(keypoints["people"]) == 1
        ), "In all pictures, we should have only one person!"

        points_2d = keypoints["people"][0]["pose_keypoints_2d"]
        assert (
            len(points_2d) == 25 * 3
        ), "We have 25 points with (x, y, c); where c is confidence."

        for point_index, (x, y, confidence) in enumerate(chunked(points_2d, 3)):
            assert x is not None, "x should be defined"
            assert y is not None, "y should be defined"
            assert confidence is not None, "confidence should be defined"
            keypoints_list.append(
                JointDescriptor(
                    x=x, y=y, confidence=confidence, joint=OpenPoseJoints(point_index)
                )
            )

        return keypoints_list


def draw_pose(
    joints: List[JointDescriptor],
    bones_of_interest: List[OpenPoseBones] = None,
    ax=None,
    showLegend=False,
    bonesWidth=1,
):
    if not bones_of_interest:
        bones_of_interest = BonesOfInterest

    if not ax:
        ax = plt.figure()

    joint_lookup = {pose.joint: pose for pose in joints}
    poses_of_interest = set()

    # Draw Bones
    for bone in bones_of_interest:
        from_joint_type, to_joint_type = bone.value
        from_joint = joint_lookup[from_joint_type]
        to_joint = joint_lookup[to_joint_type]
        poses_of_interest.add(from_joint_type)
        poses_of_interest.add(to_joint_type)
        ax.plot(
            [from_joint.x, to_joint.x], [from_joint.y, to_joint.y], linewidth=bonesWidth
        )

    # Draw Joints
    for joint_type in poses_of_interest:
        joint = joint_lookup[joint_type]
        ax.scatter(joint.x, joint.y, label=joint.joint.name)

    if showLegend:
        ax.legend()

    return ax
