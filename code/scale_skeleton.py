import numpy as np
from fractions import Fraction

# Compute the joint length with the euclidean distance between 2 joints.
def compute_joint_length(point_a, point_b):
    return np.linalg.norm(point_b - point_a)

# resize the skeleton scaling each joint
def resize_skeleton(skeleton, target_joints_length, current_joints_length):
    # Compute the ratio between the lenght we want and the current length
    scale_ratio = target_joints_length / current_joints_length
    # Calculate the scaling of the skeleton
    scaled_skeleton = skeleton * scale_ratio

    return scaled_skeleton

# Scaling the skeleton in a single frame 
def scale_single_frame(lines, skeleton):
    target_joints_length = 10

    joints_length = 0

    # Computing the total length summing the length of each joint
    for indices in lines:
        index_1, index_2 = indices
        joints_length += compute_joint_length(skeleton[index_1], skeleton[index_2])

    # Resizing the skeleton
    scaled_skeleton = resize_skeleton(skeleton, target_joints_length, joints_length)

    return scaled_skeleton

# Scaling the skeleton along all the frames
def scale_multiple_frames(lines, skeleton_frames):
    complete_scaled_skeleton = []

    # For each frame the skeleton is scaled
    for i, skeleton_frame in enumerate(skeleton_frames):
        frame = scale_single_frame(lines, skeleton_frame)
        complete_scaled_skeleton.append(frame)

    return np.array(complete_scaled_skeleton)
  
def align_pelvises(skeleton_frames_1, skeleton_frames_2):
    skeleton_aligned_complete = []

    for i, frames_1 in enumerate(skeleton_frames_1):
        # Get the pelvis position from the current frame of skeleton 1
        pelvis_position = frames_1[0] # Coordinates of the pelvis marker

        # Calculate the difference between the two pelvises
        pelvis_diff = pelvis_position - skeleton_frames_2[i][0]

        # Apply the displacement to the corresponding frame of skeleton 2
        skeleton_aligned_frame = skeleton_frames_2[i] + pelvis_diff

        # Add the aligned frame to the complete list
        skeleton_aligned_complete.append(skeleton_aligned_frame)

    # Return the aligned skeletons as a numpy array
    return np.array(skeleton_aligned_complete)     


def downsample_video(lists_of_points, lists_of_pointsCompare):
    # Calculate step size for regular removal
    lenV = len(lists_of_points)
    lenVC = len(lists_of_pointsCompare)
    if lenV == lenVC:
        return lists_of_points, lists_of_pointsCompare
    if lenV > lenVC:
        long_video = lists_of_points
        target_length = lenVC
    else:
        long_video = lists_of_pointsCompare
        target_length = lenV

    indices = np.linspace(0, len(long_video)-1, target_length, dtype=int)
    long_video = [long_video[i] for i in indices]

    if lenV > lenVC:
        return long_video, lists_of_pointsCompare
    else:
        return lists_of_points, long_video

def compute_performance(vertices, verticesCompare, bonesList):
    
    # Compute the euclidean distance between the two skeletons, for each frame and for each bone
    distances = np.array([
        np.linalg.norm(vertices[i] - verticesCompare[i], axis=1)
        for i in range(len(vertices))
    ])
    
    # We make three groups of distances: arms, legs and other
    arms_distances = []
    legs_distances = []
    other_distances = []
    
    for distList in distances:
        arms_distances.append(distList[bonesList.index('RightArm')])
        arms_distances.append(distList[bonesList.index('LeftArm')])
        arms_distances.append(distList[bonesList.index('RightForeArm')])
        arms_distances.append(distList[bonesList.index('LeftForeArm')])
        arms_distances.append(distList[bonesList.index('RightForeArmRoll')])
        arms_distances.append(distList[bonesList.index('LeftForeArmRoll')])
        arms_distances.append(distList[bonesList.index('RightHand')])
        arms_distances.append(distList[bonesList.index('LeftHand')])
        
        legs_distances.append(distList[bonesList.index('RightUpLeg')])
        legs_distances.append(distList[bonesList.index('LeftUpLeg')])
        legs_distances.append(distList[bonesList.index('RightLeg')])
        legs_distances.append(distList[bonesList.index('LeftLeg')])
        legs_distances.append(distList[bonesList.index('RightFoot')])
        legs_distances.append(distList[bonesList.index('LeftFoot')])
        legs_distances.append(distList[bonesList.index('RightToeBase')])
        legs_distances.append(distList[bonesList.index('LeftToeBase')])
        
        other_distances.append(distList[bonesList.index('Hips')])
        other_distances.append(distList[bonesList.index('Spine1')])
        other_distances.append(distList[bonesList.index('Spine2')])
        other_distances.append(distList[bonesList.index('Spine')])
        other_distances.append(distList[bonesList.index('Neck')])
        other_distances.append(distList[bonesList.index('Head')])
        other_distances.append(distList[bonesList.index('RightShoulder')])
        other_distances.append(distList[bonesList.index('LeftShoulder')])
    
    arms_metric = np.sum(arms_distances) * 1.5
    legs_metric = np.sum(legs_distances)
    other_metric = np.sum(other_distances) * 0.5
    overall_metric = arms_metric + legs_metric + other_metric
    
    
    if True:
        pass
    
    return distances


