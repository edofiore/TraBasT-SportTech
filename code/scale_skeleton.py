import numpy as np
from fractions import Fraction
import math

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
    
def compute_angle(skeleton_frames, idx_1, idx_2, idx_3, isMin=True):
    # # Indices of the RightArm and RightForeArm in the bones list
    # right_arm_idx = bonesList.index("RightArm")
    # right_forearm_idx = bonesList.index("RightForeArm")

     # Store the angles for each frame
    angles = []

    for frame in skeleton_frames:
        # Get the points for RightArm and RightForeArm
        point_segment_1_start = frame[idx_1]
        point_segment_1_end = frame[idx_2]
        point_segment_2_end = frame[idx_3]  # Assume bones are connected sequentially

        # Vectors representing the two bones
        vector_segment_1 = point_segment_1_end - point_segment_1_start
        vector_segment_2 = point_segment_2_end - point_segment_1_end

        # Compute the angle between the two vectors
        dot_product = np.dot(vector_segment_1, vector_segment_2)
        norm_product = np.linalg.norm(vector_segment_1) * np.linalg.norm(vector_segment_2)
        angle = math.degrees(math.acos(dot_product / norm_product))  # Angle in degrees
        
        # Convert to the internal angle
        internal_angle = 180 - angle  # Internal angle

        angles.append(internal_angle)
    
    angle = 0
    if(isMin):
        # Find the frame with the minimum angle
        angle = min(angles)
    else: 
        # Find the frame with the maximum angle
        angle = max(angles)
    best_frame_idx = angles.index(angle)
    
    return best_frame_idx, angle

def compute_joint_angle_differences(vertices, verticesCompare, joint_parts, is_Min=True, joint_types=None):
    """
    Calculates the difference between the minimum or maximum angles for various body segments (elbows, knees, hips, arms, etc.)
    Optionally, calculates the mean difference for specific joint types.

    Parameters:
    - vertices: the original skeleton (vertex frames)
    - verticesCompare: the compared skeleton (vertex frames)
    - joint_parts: dictionary of joints and the bones involved in them
    - is_Min: if True, calculates the minimum angles; if False, calculates the maximum angles
    - joint_types: a list of joint types to compute the mean for (e.g., ["elbows", "knees"]); if None, calculates the mean for all joints
    
    Returns:
    - results: a dictionary with the angle differences for each body segment and the conditional mean difference.
    """
    results_diff = {}
    results_angles = {}
    
    # Joint categories (types) for grouping
    joint_categories = {
        "elbows": ["RightElbow", "LeftElbow"],
        "knees": ["RightKnee", "LeftKnee"],
        "pelvis": ["RightHip", "LeftHip"],
        "arms": ["RightArm", "LeftArm"],
    }

    # Iterate through all the joints and compute the angle differences
    for joint_name, (idx_1, idx_2, idx_3) in joint_parts.items():
        # Compute the minimum or maximum angle for the first skeleton
        _, angle_1 = compute_angle(vertices, idx_1, idx_2, idx_3, isMin=is_Min)
        
        # Compute the minimum or maximum angle for the second skeleton (the compared one)
        _, angle_2 = compute_angle(verticesCompare, idx_1, idx_2, idx_3, isMin=is_Min)
        
        # Compute the absolute difference between the angles
        angle_diff = abs(angle_2 - angle_1)

        # Save the angle in the results_angles dictionary
        results_angles[f"{joint_name}_angles"] = (angle_1, angle_2)
        
        # Add the difference to the results dictionary
        results_diff[f"{joint_name}_diff"] = angle_diff
    
    # If joint_types is specified, compute the mean for those specific joint types
    if joint_types:
        for joint_type in joint_types:
            if joint_type in joint_categories:
                # Collect all angle differences for the selected joint type
                selected_joints = joint_categories[joint_type]
                selected_diffs = [results_diff[joint] for joint in selected_joints if joint in results_diff]
                
                # Compute the mean difference for the selected joint type
                if selected_diffs:
                    mean_diff = sum(selected_diffs) / len(selected_diffs)
                    results_diff[f"{joint_type}_mean_diff"] = mean_diff

    # If joint_types is not specified, compute the overall mean difference
    if not joint_types:
        mean_diff = sum(results_diff.values()) / len(results_diff)
        results_diff["mean_diff"] = mean_diff

    return results_diff, results_angles


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

    ### Compute the minimum angles and the corresponding frame for both skeletons

    # ## Compute elbows angle
    # # Compute right elbows angles
    # _, min_right_elbow_angle = compute_angle(vertices, bonesList.index('RightArm'), bonesList.index("RightForeArm"), bonesList.index("RightForeArmRoll"))
    # _, min_rigth_elbow_angle_GS = compute_angle(verticesCompare, bonesList.index('RightArm'), bonesList.index("RightForeArm"), bonesList.index("RightForeArmRoll"))
    # # Compute the difference in the minimum angles
    # min_right_elbow_angle_diff = abs(min_rigth_elbow_angle_GS - min_right_elbow_angle)

    # # Compute left elbows angles
    # _, min_left_elbow_angle = compute_angle(vertices, bonesList.index('LeftArm'), bonesList.index("LeftForeArm"), bonesList.index('LeftForeArmRoll'))
    # _, min_left_elbow_angle_GS = compute_angle(verticesCompare, bonesList.index('LeftArm'), bonesList.index("LeftForeArm"), bonesList.index('LeftForeArmRoll'))
    # # Compute the difference in the minimum angles
    # min_left_elbows_angles_diff = abs(min_left_elbow_angle_GS - min_left_elbow_angle)

    # ## Compute min arms angles
    # # Compute right arms angles
    # _, min_right_arm_angle = compute_angle(vertices, bonesList.index('Spine'), bonesList.index("RightArm"), bonesList.index('RightForeArm'))
    # _, min_right_arm_angle_GS = compute_angle(verticesCompare, bonesList.index('Spine'), bonesList.index("RightArm"), bonesList.index('RightForeArm'))
    # # Compute the difference in the minimum angles
    # min_right_arms_angles_diff = abs(min_right_arm_angle_GS - min_right_arm_angle)
  

    # # Compute left arms angles
    # _, min_left_arm_angle = compute_angle(vertices, bonesList.index('Spine'), bonesList.index("LeftArm"), bonesList.index('LeftForeArm'))
    # _, min_left_arm_angle_GS = compute_angle(verticesCompare, bonesList.index('Spine'), bonesList.index("LeftArm"), bonesList.index('LeftForeArm'))
    # # Compute the difference in the minimum angles
    # min_left_arms_angles_diff = abs(min_left_arm_angle_GS - min_left_arm_angle)
    
    # mean_min_arms_angles_diff =  (min_right_arms_angles_diff + min_left_arms_angles_diff) / 2

    # ## Compute min knees angles
    # # Compute right knees angles
    # _, min_right_knee_angle = compute_angle(vertices, bonesList.index('RightUpLeg'), bonesList.index("RightLeg"), bonesList.index('RightFoot'))
    # _, min_right_knee_angle_GS = compute_angle(verticesCompare, bonesList.index('RightUpLeg'), bonesList.index("RightLeg"), bonesList.index('RightFoot'))
    # # Compute the difference in the minimum angles
    # min_right_knees_angles_diff = abs(min_right_knee_angle_GS - min_right_knee_angle)

    # # Compute left knees angles
    # _, min_left_knee_angle = compute_angle(vertices, bonesList.index('LeftUpLeg'), bonesList.index("LeftLeg"), bonesList.index('LeftFoot'))
    # _, min_left_knee_angle_GS = compute_angle(verticesCompare, bonesList.index('LeftUpLeg'), bonesList.index("LeftLeg"), bonesList.index('LeftFoot'))
    # # Compute the difference in the minimum angles
    # min_left_knees_angles_diff = abs(min_left_knee_angle_GS - min_left_knee_angle)

    # mean_min_knees_angles_diff =  (min_right_knees_angles_diff + min_left_knees_angles_diff) / 2

    # ## Compute min pelvis angles
    # # Compute right hips angles
    # _, min_right_hip_angle = compute_angle(vertices, bonesList.index('Spine'), bonesList.index('RightUpLeg'), bonesList.index("RightLeg"))
    # _, min_right_hip_angle_GS = compute_angle(verticesCompare, bonesList.index('Spine'), bonesList.index('RightUpLeg'), bonesList.index("RightLeg"))
    # # Compute the difference in the minimum angles
    # min_right_hips_angles_diff = abs(min_right_hip_angle_GS - min_right_hip_angle)

    # # Compute left hips angles
    # _, min_left_hip_angle = compute_angle(vertices, bonesList.index('Spine'), bonesList.index('LeftUpLeg'), bonesList.index("LeftLeg"))
    # _, min_left_hip_angle_GS = compute_angle(verticesCompare, bonesList.index('Spine'), bonesList.index('LeftUpLeg'), bonesList.index("LeftLeg"))
    # # Compute the difference in the minimum angles
    # min_left_hips_angles_diff = abs(min_left_hip_angle_GS - min_left_hip_angle)

    # mean_min_hips_angles_diff =  (min_right_hips_angles_diff + min_left_hips_angles_diff) / 2

    
    # ### Compute the maximum angles for both skeletons
    # ## Compute max arms angles
    # # Compute right arms angles
    # _, max_right_arm_angle = compute_angle(vertices, bonesList.index('Spine'), bonesList.index("RightArm"), bonesList.index('RightForeArm'), isMin=False)
    # _, max_right_arm_angle_GS = compute_angle(verticesCompare, bonesList.index('Spine'), bonesList.index("RightArm"), bonesList.index('RightForeArm'), isMin=False)
    # # Compute the difference in the minimum angles
    # max_right_arms_angles_diff = abs(max_right_arm_angle_GS - max_right_arm_angle)

    # # Compute left arms angles
    # _, max_left_arm_angle = compute_angle(vertices, bonesList.index('Spine'), bonesList.index("LeftArm"), bonesList.index('LeftForeArm'), isMin=False)
    # _, max_left_arm_angle_GS = compute_angle(verticesCompare, bonesList.index('Spine'), bonesList.index("LeftArm"), bonesList.index('LeftForeArm'), isMin=False)
    # # Compute the difference in the minimum angles
    # max_left_arms_angles_diff = abs(max_left_arm_angle_GS - max_left_arm_angle)

    # mean_max_arms_angles_diff =  (max_right_arms_angles_diff + max_left_arms_angles_diff) / 2


    # Define the body segments with the corresponding bone indices
    joint_parts_min = {
        "RightElbow": (bonesList.index('RightArm'), bonesList.index("RightForeArm"), bonesList.index("RightForeArmRoll")),
        "LeftElbow": (bonesList.index('LeftArm'), bonesList.index("LeftForeArm"), bonesList.index('LeftForeArmRoll')),
        "RightArm": (bonesList.index("Spine"), bonesList.index('RightArm'), bonesList.index("RightForeArm")),
        "LeftArm": (bonesList.index('Spine'), bonesList.index('LeftArm'), bonesList.index("LeftForeArm")),
        "RightKnee": (bonesList.index('RightUpLeg'), bonesList.index("RightLeg"), bonesList.index('RightFoot')),
        "LeftKnee": (bonesList.index('LeftUpLeg'), bonesList.index("LeftLeg"), bonesList.index('LeftFoot')),
        "RightHip": (bonesList.index('Spine'), bonesList.index('RightUpLeg'), bonesList.index("RightLeg")),
        "LeftHip": (bonesList.index('Spine'), bonesList.index('LeftUpLeg'), bonesList.index("LeftLeg")),
    }

    # Call the function to get min_angle differences and the mean for elbows
    results_min_diff, results_min = compute_joint_angle_differences(vertices, verticesCompare, joint_parts_min, is_Min=True, joint_types=["knees", "pelvis", "arms"])

    # Print the results
    print("MINIMUM ANGLES")
    print(results_min)
    print(results_min_diff)

    # Define the body segments with the corresponding bone indices
    joint_parts_max = {
        "RightArm": (bonesList.index("Spine"), bonesList.index('RightArm'), bonesList.index("RightForeArm")),
        "LeftArm": (bonesList.index('Spine'), bonesList.index('LeftArm'), bonesList.index("LeftForeArm")),
    }

    # Call the function to get min_angle differences and the mean for elbows
    results_max_diff, results_max = compute_joint_angle_differences(vertices, verticesCompare, joint_parts_max, is_Min=False, joint_types=["arms"])

    # Print the results
    print("MAXIMUM ANGLES")
    print(results_max)
    print(results_max_diff)



    if True:
        pass
    
    return distances

# def compute_angle(x1, y1, x2, y2):

#     if (x2 - x1) != 0:
#         slope1 = (y2 - y1) / (x2 - x1)
#     elif (x2 - x1 )== 0:
#         slope1 = math.inf

#     angle1 = math.degrees(math.atan(slope1))

#     angle_diff = abs(90 - angle1)

#     return angle_diff

