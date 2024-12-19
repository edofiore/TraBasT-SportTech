import numpy as np

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
    for indices in lines:
        index_1, index_2 = indices
        joints_length += compute_joint_length(skeleton[index_1], skeleton[index_2])

    scaled_skeleton = resize_skeleton(skeleton, target_joints_length, joints_length)

    return scaled_skeleton

# Scaling the skeleton along all the frames
def scale_multiple_frames(lines, skeleton_points):
    complete_scaled_skeleton = []
    for i, data in enumerate(skeleton_points):
        frame = scale_single_frame(lines, data)
        complete_scaled_skeleton.append(frame)

    return complete_scaled_skeleton
  
