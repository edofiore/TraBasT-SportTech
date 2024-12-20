import numpy as np
from fractions import Fraction
import math

# LIMITS
ARMS_LIMIT = 350
LEGS_LIMIT = 250
OTHER_LIMIT = 140
OVERALL_LIMIT = 710
SPEED_LIMIT = 40

# METRICS
ARMS_METRIC = 1.4
LEGS_METRIC = 1.1
OTHER_METRIC = 0.5
SPEED_METRIC = 3


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

# Align the pelvises of the two skeletons to have a better comparison
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

# Downsample the video or the gold standard to have the same number of frames
def downsample_video(lists_of_points, lists_of_pointsCompare):
    # Calculate step size for regular removal
    lenP = len(lists_of_points)
    lenPC = len(lists_of_pointsCompare)
    if lenP == lenPC:
        return lists_of_points, lists_of_pointsCompare, lenP, lenPC
    if lenP > lenPC:
        long_video = lists_of_points
        target_length = lenPC
    else:
        long_video = lists_of_pointsCompare
        target_length = lenP

    indices = np.linspace(0, len(long_video)-1, target_length, dtype=int)
    long_video = [long_video[i] for i in indices]

    if lenP > lenPC:
        return long_video, lists_of_pointsCompare, lenP, lenPC
    else:
        return lists_of_points, long_video, lenP, lenPC
    
def compute_angle(skeleton_frames, idx_1, idx_2, idx_3, is_min=True):
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
    if(is_min):
        # Find the frame with the minimum angle
        angle = min(angles)
    else: 
        # Find the frame with the maximum angle
        angle = max(angles)
    best_frame_idx = angles.index(angle)
    
    return best_frame_idx, angle

def compute_joint_angle_differences(vertices, vertices_compare, joint_parts, is_min=True, joint_types=None):
    """
    Calculates the difference between the minimum or maximum angles for various body segments (elbows, knees, hips, arms, etc.)
    Optionally, calculates the mean difference for specific joint types.

    Parameters:
    - vertices: the original skeleton (vertex frames)
    - vertices_compare: the compared skeleton (vertex frames)
    - joint_parts: dictionary of joints and the bones involved in them
    - is_min: if True, calculates the minimum angles; if False, calculates the maximum angles
    - joint_types: a list of joint types to compute the mean for (e.g., ["Elbows", "Knees"]); if None, calculates the mean for all joints

    """
    results_diff = {}
    results_angles = {}
    
    # Joint categories (types) for grouping
    joint_categories = {
        "Elbows": ["RightElbow", "LeftElbow"],
        "Knees": ["RightKnee", "LeftKnee"],
        "Pelvis": ["RightHip", "LeftHip"],
        "Arms": ["RightArm", "LeftArm"],
    }

    # Iterate through all the joints and compute the angle differences
    for joint_name, (idx_1, idx_2, idx_3) in joint_parts.items():
        # Compute the minimum or maximum angle for the first skeleton
        _, angle_1 = compute_angle(vertices, idx_1, idx_2, idx_3, is_min=is_min)
        
        # Compute the minimum or maximum angle for the second skeleton (the compared one)
        _, angle_2 = compute_angle(vertices_compare, idx_1, idx_2, idx_3, is_min=is_min)
        
        # Compute the absolute difference between the angles
        angle_diff = abs(angle_2 - angle_1)

        # Save the angle in the results_angles dictionary
        results_angles[f"{joint_name}_angles"] = (angle_1, angle_2)
        
        # Add the difference to the results dictionary
        results_diff[f"{joint_name}_diff"] = angle_diff
    
   # Compute the mean for individual joint categories (like elbows, knees, etc.)
    category_mean_diffs = {}
    for joint_type, joint_names in joint_categories.items():
        selected_diffs = [results_diff.get(f"{joint_name}_diff", 0) for joint_name in joint_names]
        
        # Compute the mean difference for the selected joint type
        if selected_diffs:
            mean_diff = sum(selected_diffs) / len(selected_diffs)
            category_mean_diffs[f"{joint_type}_mean_diff"] = mean_diff
    
    # Add the category mean differences to the results
    results_diff.update(category_mean_diffs)

    # If joint_types is specified, compute the mean for those specific joint types
    if joint_types:
        for joint_type in joint_types:
            if joint_type in joint_categories:
                # Collect all angle differences for the selected joint type
                selected_joints = joint_categories[joint_type]
                selected_diffs = [results_diff.get(f"{joint_name}_diff", 0) for joint_name in selected_joints]
                
                # Compute the mean difference for the selected joint type
                if selected_diffs:
                    mean_diff = sum(selected_diffs) / len(selected_diffs)
                    results_diff[f"{joint_type}_mean_diff"] = mean_diff

    # If joint_types is not specified, compute the overall mean difference
    total_mean_diff = sum(results_diff.values()) / len(results_diff)
    results_diff["Total_mean_diff"] = total_mean_diff

    return results_diff, results_angles


# Compute the performance of the shooting with respect to the gold standard
def compute_performance(vertices, vertices_compare, bones_list, len_p, len_p_compare):
    
    # Compute the euclidean distance between the two skeletons, for each frame and for each bone
    distances = np.array([
        np.linalg.norm(vertices[i] - vertices_compare[i], axis=1)
        for i in range(len(vertices))
    ])
    
    # We make three groups of distances: arms, legs and other
    arms_distances = []
    legs_distances = []
    other_distances = []
    
    for dist_list in distances:
        arms_distances.append(dist_list[bones_list.index('RightArm')])
        arms_distances.append(dist_list[bones_list.index('LeftArm')])
        arms_distances.append(dist_list[bones_list.index('RightForeArm')])
        arms_distances.append(dist_list[bones_list.index('LeftForeArm')])
        arms_distances.append(dist_list[bones_list.index('RightForeArmRoll')])
        arms_distances.append(dist_list[bones_list.index('LeftForeArmRoll')])
        arms_distances.append(dist_list[bones_list.index('RightHand')])
        arms_distances.append(dist_list[bones_list.index('LeftHand')])
        
        legs_distances.append(dist_list[bones_list.index('RightUpLeg')])
        legs_distances.append(dist_list[bones_list.index('LeftUpLeg')])
        legs_distances.append(dist_list[bones_list.index('RightLeg')])
        legs_distances.append(dist_list[bones_list.index('LeftLeg')])
        legs_distances.append(dist_list[bones_list.index('RightFoot')])
        legs_distances.append(dist_list[bones_list.index('LeftFoot')])
        legs_distances.append(dist_list[bones_list.index('RightToeBase')])
        legs_distances.append(dist_list[bones_list.index('LeftToeBase')])
        
        other_distances.append(dist_list[bones_list.index('Hips')])
        other_distances.append(dist_list[bones_list.index('Spine1')])
        other_distances.append(dist_list[bones_list.index('Spine2')])
        other_distances.append(dist_list[bones_list.index('Spine')])
        other_distances.append(dist_list[bones_list.index('Neck')])
        other_distances.append(dist_list[bones_list.index('Head')])
        other_distances.append(dist_list[bones_list.index('RightShoulder')])
        other_distances.append(dist_list[bones_list.index('LeftShoulder')])
    
    speed_diff = len_p - len_p_compare
    speed = 'good'
    suggestion = 'keep this speed'
    if speed_diff <= -SPEED_LIMIT:
        speed = 'a little fast'
        suggestion = 'slow down a little bit'
    elif speed_diff >= SPEED_LIMIT:
        speed = 'a little slow'
        suggestion = 'speed up a little bit'


    ### Compute the angles
    # Define the body segments with the corresponding bone indices
    joint_parts_min = {
        "RightElbow": (bones_list.index('RightArm'), bones_list.index("RightForeArm"), bones_list.index("RightForeArmRoll")),
        "LeftElbow": (bones_list.index('LeftArm'), bones_list.index("LeftForeArm"), bones_list.index('LeftForeArmRoll')),
        "RightArm": (bones_list.index("Spine"), bones_list.index('RightArm'), bones_list.index("RightForeArm")),
        "LeftArm": (bones_list.index('Spine'), bones_list.index('LeftArm'), bones_list.index("LeftForeArm")),
        "RightKnee": (bones_list.index('RightUpLeg'), bones_list.index("RightLeg"), bones_list.index('RightFoot')),
        "LeftKnee": (bones_list.index('LeftUpLeg'), bones_list.index("LeftLeg"), bones_list.index('LeftFoot')),
        "RightHip": (bones_list.index('Spine'), bones_list.index('RightUpLeg'), bones_list.index("RightLeg")),
        "LeftHip": (bones_list.index('Spine'), bones_list.index('LeftUpLeg'), bones_list.index("LeftLeg")),
    }

    # Call the function to get min_angle differences and the mean for elbows
    results_min_diff, results_min_angles = compute_joint_angle_differences(vertices, vertices_compare, joint_parts_min, is_min=True, joint_types=["Knees", "Pelvis", "Arms"])

    # Print the results
    # print("MINIMUM ANGLES")
    # print(f"Minimum angles: {results_min_angles}")
    # print(f"Minimum angles diff: {results_min_diff}")

    # Define the body segments with the corresponding bone indices
    joint_parts_max = {
        'RightArm': (bones_list.index('Spine'), bones_list.index('RightArm'), bones_list.index('RightForeArm')),
        'LeftArm': (bones_list.index('Spine'), bones_list.index('LeftArm'), bones_list.index('LeftForeArm')),
    }

    # Call the function to get min_angle differences and the mean for elbows
    results_max_diff, results_max_angles = compute_joint_angle_differences(vertices, vertices_compare, joint_parts_max, is_min=False, joint_types=['Arms', 'Pelvis'])

    # Print the results
    # print("MAXIMUM ANGLES")
    # print(f"Maximum angles: {results_max_angles}")
    # print(f"Maximum angles diff: {results_max_diff}")

    
    speed_diff = abs(speed_diff) * SPEED_METRIC
    arms_angles_metric = results_min_diff['RightElbow_diff'] + results_min_diff['LeftElbow_diff'] + results_min_diff['Arms_mean_diff'] + results_max_diff['Arms_mean_diff']
    arms_metric = (np.sum(arms_distances) + arms_angles_metric) * ARMS_METRIC
    legs_angles_metric = results_min_diff['Pelvis_mean_diff'] + results_max_diff['Pelvis_mean_diff'] + results_min_diff['Knees_mean_diff']
    legs_metric = (np.sum(legs_distances) + legs_angles_metric) * LEGS_METRIC
    other_metric = np.sum(other_distances) * OTHER_METRIC
    overall_metric = arms_metric + legs_metric + other_metric + speed_diff


    print(f'Total arms: {arms_angles_metric}')
    print(f'Total legs: {arms_angles_metric}')
    print(f'Arms metric: {arms_metric}')
    print(f'Legs metric: {legs_metric}')
    print(f'Other metric: {other_metric}')
    print(f'Overall metric: {overall_metric}')
    
    if overall_metric < OVERALL_LIMIT:
        print('Your shooting is good, keep it up!')
    else:
        print('You should adjust a little your shooting, but do not be discouraged! I will help you')
        if arms_metric > ARMS_LIMIT:
            print('Your should adjust a little your arms')
        if legs_metric > LEGS_LIMIT:
            print('Your should adjust a little your legs')
        if other_metric > OTHER_LIMIT:
            print('Your should adjust a little your back and your movement as a whole')
        
    print(f'Your shooting is {speed}, try to {suggestion}')

