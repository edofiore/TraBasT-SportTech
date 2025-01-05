import numpy as np
from fractions import Fraction
import math
import matplotlib.pyplot as plt
from types import SimpleNamespace

# ----------------------------
## LEGEND
# - GS: Gold Standard
# ----------------------------


### Define threshold for the different parts of the body
# # LIMITS
# ARMS_LIMIT = 460
# LEGS_LIMIT = 260 
# OTHER_LIMIT = 140
# OVERALL_LIMIT = 900
# SPEED_LIMIT = 40

# # ANGLES LIMITS
# # The LIMITS consider the 2 arms/elbows/knees together
# ELBOWS_ANGLE_LIMIT = 20 # Limit to know if the elbow is too bent or too extended
# ARMS_ANGLE_LIMIT = 30 # Limit only to know if the arms are too high or too low
# # if the arms range is too large means that the shot is too irregular, because there is a moment during the shot
# # where your arms are too low, and another where your arms are too much over/behind the head/neck
# # ARMS_R_RANGE_LIMIT = 130 # Range (max_R_arm - min_R_arm)
# # ARMS_L_RANGE_LIMIT = 30 # Range (max_L_arm - min_L_arm)
# KNEES_ANGLE_LIMIT = 15
# PELVIS_ANGLE_LIMIT = 3


# Define threshold for the different parts of the body
LIMITS = {
    'ARMS_LIMIT': SimpleNamespace(value=460, importance=3),   # Maximum limit for arms
    'LEGS_LIMIT': SimpleNamespace(value=260, importance=2),   # Maximum limit for legs
    'OTHER_LIMIT': SimpleNamespace(value=140, importance=1),  # Maximum limit for posture
    'OVERALL_LIMIT': SimpleNamespace(value=900, importance=5),  # Overall performance limit
    'SPEED_LIMIT': SimpleNamespace(value=40, importance=2),   # Limit for shooting speed
}

# Define the limits for angles: accectable difference from the GS
ANGLES_LIMITS = {
    'R_ELBOW_ANGLE_LIMIT': SimpleNamespace(value=20, importance=4),  # Limit in order to know if the right elbow is too bent or too extended
    'L_ELBOW_ANGLE_LIMIT': SimpleNamespace(value=20, importance=4),  # Limit in order to know if the left elbow is too bent or too extended
    'ARMS_ANGLE_LIMIT': SimpleNamespace(value=30, importance=2),  # Limit in order to know if the arms are too high or too low
    'R_ARM_RANGE_ANGLE_LIMIT': SimpleNamespace(value=30, importance=2),  # Limit in order to know if the movement of the right arm is too wide
    'L_ARM_RANGE_ANGLE_LIMIT': SimpleNamespace(value=30, importance=2),  # Limit in order to know if the movement of the left arm is too wide
    'KNEES_ANGLE_LIMIT': SimpleNamespace(value=15, importance=2),  # Limit for knee bend during shooting
    'PELVIS_ANGLE_LIMIT': SimpleNamespace(value=3, importance=1),  # Limit for pelvis position
}

### Define coefficients to apply at the different parts of the body
## They are choose empirically
# METRICS
ARMS_METRIC = 1.4   # The arms have a lot of importance in a free throw
LEGS_METRIC = 1.1   # The legs are important, but less so than arms
OTHER_METRIC = 0.5  # It has low importance in the free throw
SPEED_METRIC = 3    # The speed of the free throw has a relevant importance


# Compute the joint length with the euclidean distance between 2 joints.
def compute_joint_length(point_a, point_b):
    return np.linalg.norm(point_b - point_a)

# resize the skeleton scaling each joint
def resize_skeleton(skeleton, target_joints_length, current_joints_length):

    """
    Parameters:
    - skeleton: list of points of the skeleton in a single frame
    - target_joints_length: length we want
    - current_joints_length: current total length
    """

    # Compute the ratio between the length we want and the current length
    scale_ratio = target_joints_length / current_joints_length
    # Calculate the scaling of the skeleton
    scaled_skeleton = skeleton * scale_ratio

    return scaled_skeleton

# Scaling the skeleton in a single frame 
def scale_single_frame(edges, skeleton):

    """
    Parameters:
    - edges: edges in the graph
    - skeleton: list of points of the skeleton in a single frame
    """
    
    target_joints_length = 10

    joints_length = 0

    # Computing the total length summing the length of each joint
    for indices in edges:
        index_1, index_2 = indices
        joints_length += compute_joint_length(skeleton[index_1], skeleton[index_2])

    # Resizing the skeleton
    scaled_skeleton = resize_skeleton(skeleton, target_joints_length, joints_length)

    return scaled_skeleton

# Scaling the skeleton along all the frames
def scale_multiple_frames(edges, skeleton_frames):

    """
    Parameters:
    - edges: edges in the graph
    - skeleton_frames: list of points of the skeleton at each frame
    """

    complete_scaled_skeleton = []

    # For each frame the skeleton is scaled
    for i, skeleton_frame in enumerate(skeleton_frames):
        frame = scale_single_frame(edges, skeleton_frame)
        complete_scaled_skeleton.append(frame)

    return np.array(complete_scaled_skeleton)

# Align the pelvises of the two skeletons to have a better comparison
def align_pelvises(skeleton_frames_1, skeleton_frames_2):


    """
    Parameters:
    - skeleton_frames_1: list of points of the skeleton of the first player
    - skeleton_frames_2: list of points of the skeleton of the Golden Standard to compare
    """

    skeleton_aligned_complete = []

    for i, frames_1 in enumerate(skeleton_frames_1):
        # Get the pelvis position from the current frame of skeleton 1
        pelvis_position_1 = frames_1[0] # Coordinates of the pelvis marker

        # Calculate the difference between the two pelvises
        pelvis_diff = pelvis_position_1 - skeleton_frames_2[i][0]

        # Apply the displacement to the corresponding frame of skeleton 2
        skeleton_aligned_frame = skeleton_frames_2[i] + pelvis_diff

        # Add the aligned frame to the complete list
        skeleton_aligned_complete.append(skeleton_aligned_frame)

    # Return the aligned skeletons as a numpy array
    return np.array(skeleton_aligned_complete)     

# Downsample the video or the gold standard to have the same number of frames
def downsample_video(lists_of_points, lists_of_pointsCompare):
    """
    Parameters:
    - lists_of_points: 
    - lists_of_pointsCompare: 
    """

    # Calculate step size for regular removal
    len_p = len(lists_of_points)
    len_p_c = len(lists_of_pointsCompare)
    if len_p == len_p_c:
        return lists_of_points, lists_of_pointsCompare, len_p, len_p_c
    if len_p > len_p_c:
        long_video = lists_of_points
        target_length = len_p_c
    else:
        long_video = lists_of_pointsCompare
        target_length = len_p

    indices = np.linspace(0, len(long_video)-1, target_length, dtype=int)
    long_video = [long_video[i] for i in indices]

    if len_p > len_p_c:
        return long_video, lists_of_pointsCompare, len_p, len_p_c
    else:
        return lists_of_points, long_video, len_p, len_p_c
    
# Compute the angle between 2 segments
def compute_angle(skeleton_frames, idx_1, idx_2, idx_3, is_min=True):
    """
    Parameters:
    - skeleton_frames: all the frames of the skeleton
    - idx_1, idx_2, idx_3: the 3 indices of the joints used to compute the angle
    - is_min: if True, calculates the minimum angles; if False, calculates the maximum angles
    """

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
        # vector_segment_1 = np.array(point_segment_1_end) - np.array(point_segment_1_start)
        # vector_segment_2 = np.array(point_segment_2_end) - np.array(point_segment_1_end)

        # Compute the angle between the two vectors
        dot_product = np.dot(vector_segment_1, vector_segment_2)
        norm_product = np.linalg.norm(vector_segment_1) * np.linalg.norm(vector_segment_2)
        if norm_product == 0:
            angle = 0
        else:
            angle = np.degrees(np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0)))  # Angle in degrees

        # print("Point 1:", point_segment_1_start)
        # print("Point 2:", point_segment_1_end)
        # print("Point 3:", point_segment_2_end)
        # print("Vector 1:", vector_segment_1)
        # print("Vector 2:", vector_segment_2)
        # print("Dot Product:", dot_product)
        # print("Norm Product:", norm_product)
        
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

# Calculates the difference between the minimum or maximum angles for various body segments (elbows, knees, hips, arms, etc.)
# Optionally, calculates the mean difference for specific joint types.
def compute_joint_angle_differences(vertices, vertices_compare, joint_parts, is_min=True, joint_types=None):
    """
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
        "Elbows": ["RightElbow", "LeftElbow"],  # 0 if the elbow is completely bent, 180 if the elbow is fully extended (outstretched arm)
        "Knees": ["RightKnee", "LeftKnee"], # 0 if the knee is completely bent, 180 if the knee is fully extended (the person stand up straight)
        "Pelvis": ["RightHip", "LeftHip"],  # 0 if the skeleton is completely bent on the legs, 180 if the skeleton is fully extended (the person stand up straight), > 180 if the pelvis is belt backwards
        "Arms": ["RightArm", "LeftArm"], # 0 if the arm is completely lowered, 180 if the arm is fully above the head, > 180 if the arm goes behind the head
    } 

    # Iterate through all the joints and compute the angle differences
    for joint_name, (idx_1, idx_2, idx_3) in joint_parts.items():
        # Compute the minimum or maximum angle for the first skeleton
        _, angle_1 = compute_angle(vertices, idx_1, idx_2, idx_3, is_min=is_min)
        # print(f"ANGLE_{joint_name}-1", angle_1)
        
        # Compute the minimum or maximum angle for the second skeleton (the compared one)
        _, angle_2 = compute_angle(vertices_compare, idx_1, idx_2, idx_3, is_min=is_min)
        # print(f"ANGLE_{joint_name}-2", angle_2)
        
        # Compute the absolute difference between the angles
        # angle_diff = abs(angle_2 - angle_1)
        angle_diff = angle_2 - angle_1

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


# actual_score = 0
def define_suggestions(actual_score, parameters):
    
    # actual_score = parameters["actual_score"]
    overall_metric = parameters["overall_metric"]
    arms_metric = parameters["arms_metric"]
    elbows_R_min_diff = parameters["elbows_R_min_diff"]
    elbows_L_min_diff = parameters["elbows_L_min_diff"]
    arms_mean_diff_min = parameters["arms_mean_diff_min"]
    arms_mean_diff_max = parameters["arms_mean_diff_max"]
    range_R_arm = parameters["range_R_arm"]
    range_R_arm_GS = parameters["range_R_arm_GS"]
    range_L_arm = parameters["range_L_arm"]
    range_L_arm_GS = parameters["range_L_arm_GS"]
    legs_metric = parameters["legs_metric"]
    knees_mean_diff = parameters["knees_mean_diff"]
    other_metric = parameters["other_metric"]
    pelvis_mean_diff_min = parameters["pelvis_mean_diff_min"]
    speed_diff = parameters["speed_diff"]
    
    # Check if it is a good free threshold
    if overall_metric < LIMITS['OVERALL_LIMIT'].value:
        actual_score += LIMITS['OVERALL_LIMIT'].importance
        print('\nYour shooting is good, keep it up!\n')

    else:
        print('\nYou should adjust a little your shooting, but do not be discouraged! I will help you!\n')

    # Evaluation of the player's free throw
    if arms_metric < LIMITS['ARMS_LIMIT'].value:
        actual_score += LIMITS['ARMS_LIMIT'].importance
    else:
        print('Your arm positioning could use some adjustments.')
        
        # Right elbow
        if elbows_R_min_diff > ANGLES_LIMITS['R_ELBOW_ANGLE_LIMIT'].value:
            print('→ Your right elbow is too bent. Try to extend your right elbow a little more.')
        elif elbows_R_min_diff < -ANGLES_LIMITS['R_ELBOW_ANGLE_LIMIT'].value:
            print('→ Your right elbow is too extended. Try to bend your right elbow a little more.')
        else: 
            actual_score += ANGLES_LIMITS['R_ELBOW_ANGLE_LIMIT'].importance
        
        # Left elbow
        if elbows_L_min_diff > ANGLES_LIMITS['L_ELBOW_ANGLE_LIMIT'].value:
            print('→ Your left elbow is too bent. Try to extend your left elbow a little more.')
        elif elbows_L_min_diff < -ANGLES_LIMITS['L_ELBOW_ANGLE_LIMIT'].value:
            print('→ Your left elbow is too extended. Try to bend your left elbow a little more.')
        else: 
            actual_score += ANGLES_LIMITS['L_ELBOW_ANGLE_LIMIT'].importance

        # Arm angles during the preparation
        if (arms_mean_diff_min > ANGLES_LIMITS['ARMS_ANGLE_LIMIT'].value) or (arms_mean_diff_max < -LIMITS['ARMS_ANGLE_LIMIT'].value):
            print('→ Your arms are too low. Try to raise them slightly.')
        elif (arms_mean_diff_min < -ANGLES_LIMITS['ARMS_ANGLE_LIMIT'].value) or (arms_mean_diff_max > LIMITS['ARMS_ANGLE_LIMIT'].value):
            print('→ Your arms are too high. Try to lower them slightly.')
        else:
            actual_score += ANGLES_LIMITS['ARMS_ANGLE_LIMIT'].importance

        # Arm movement range checks
        if range_R_arm > range_R_arm_GS + ANGLES_LIMITS['R_ARM_RANGE_ANGLE_LIMIT'].value:
            print('→ Your right arm movement is uncoordinated and too spread out. Try to maintain a smoother and more balanced motion.')
        else:
            actual_score += ANGLES_LIMITS['R_ARM_RANGE_ANGLE_LIMIT'].importance
        
        if range_L_arm > range_L_arm_GS + ANGLES_LIMITS['L_ARM_RANGE_ANGLE_LIMIT'].value:
            print('→ Your left arm movement is inconsistent and too spread out. Try to maintain a smoother and more balanced motion.')
        else:
            actual_score += ANGLES_LIMITS['L_ARM_RANGE_ANGLE_LIMIT'].importance


    # Legs adjustments
    if legs_metric < LIMITS['LEGS_LIMIT'].value:
        actual_score += LIMITS['LEGS_LIMIT'].importance
    else:
        print('\nLeg adjustments needed:')
        print('Your legs could use some adjustments. Try the following:')
        
        if knees_mean_diff > ANGLES_LIMITS['KNEES_ANGLE_LIMIT'].value:
            print('→ Your posture is too straight. Try to bend your knees a little more.')
        elif knees_mean_diff < -ANGLES_LIMITS['KNEES_ANGLE_LIMIT'].value:
            print('→ Your knees are too bent. Try to extend your knees a little more.')
        else:
            actual_score += ANGLES_LIMITS['KNEES_ANGLE_LIMIT'].importance


    # Back and overall posture
    if other_metric < LIMITS['OTHER_LIMIT'].value:
        actual_score += LIMITS['OTHER_LIMIT'].importance
    else:
        print('\nBack and posture adjustments needed:')
        print('Your overall posture could use some adjustments. Try the following:')
        
        # We should use this? It has sense?
        if pelvis_mean_diff_min > ANGLES_LIMITS['PELVIS_ANGLE_LIMIT'].value:
            print('→ Your posture is too rigid. Try bending forward slightly at the pelvis.')
        elif pelvis_mean_diff_min < -ANGLES_LIMITS['PELVIS_ANGLE_LIMIT'].value:
            print('→ Your posture is too bent forward. Try to straighten your pelvis a little.')
        else:
            actual_score += ANGLES_LIMITS['PELVIS_ANGLE_LIMIT'].importance

    speed = 'good'
    suggestion = 'keep this speed'
    if speed_diff <= -LIMITS['SPEED_LIMIT'].value:
        speed = 'a little too fast'
        suggestion = 'slow down a little bit'
    elif speed_diff >= LIMITS['SPEED_LIMIT'].value:
        speed = 'a little too slow'
        suggestion = 'speed up a little bit'
    else:
        actual_score += LIMITS['SPEED_LIMIT'].importance

    # Speed suggestion
    print('\nAbout the speed:')
    print(f'Your shooting speed is {speed}. Try to {suggestion} for a more effective shot.\n')

    return actual_score


    # percentage = calculate_total_percentage(actual_score, LIMITS)
    # print(f'TOTAL GOODNESS: {percentage}%.')

    # if percentage <= 40:
    #     print("BEGINNER: You're a beginner. Focus on the basics.\n")
    # elif percentage <= 60:
    #     print("AMATEUR PLAYER: The basics are set, now refine your technique.\n")
    # elif percentage <= 80:
    #     print("PRO PLAYER: Solid technique! Just fine-tune your shot.\n")
    # elif percentage <= 90:
    #     print("ELITE PLAYER: Excellent! You're almost at the top.\n")
    # else:
    #     print("GOAT PLAYER: Unbelievable! Your form matches the Gold Standard.\n")



    # -----------------------------------------------------------------------------------------
    ### OLD VERSION

    #     if arms_metric > ARMS_LIMIT:
    #         print('You should adjust a little your arms')
    #         # If the difference of the elbows angles with the GS is over the acceptable limit (TOO BENT), suggest increasing the elbow angles
    #         # If the elbow is fully extended we have 180°
    #         if elbows_R_min_diff > ELBOWS_ANGLE_LIMIT:
    #              print('Your right elbow is too bent. Try to extend your right elbow a little more.')
    #         # if it is below the acceptable limit, suggest increasing the elbow angles
    #         elif elbows_R_min_diff < -ELBOWS_ANGLE_LIMIT:
    #             print('Your right elbow is too extended. Try to bend your right elbow a little more.')

    #         if elbows_L_min_diff > ELBOWS_ANGLE_LIMIT:
    #              print('Your left elbow is too bent. Try to extend your left elbow a little more.')
    #         # if it is below the acceptable limit, suggest increasing the elbow angles
    #         elif elbows_L_min_diff < -ELBOWS_ANGLE_LIMIT:
    #             print('Your left elbow is too extended. Try to bend your left elbow a little more.')
            


    #         ### At the moment could be the same beacause we don't know when the min and max point are reached
    #         ## ??? WE COULD USE THE FRAME WHEN THE POINTS ARE REACHED ???
    #         # FIRST PART OF THE MOVEMENT
    #         # Hypothetically this is during the preparation of the free throw
    #         # Check if the mean difference with the GS exceeds the acceptable limits for arm angles,
    #         # if it is greater than the positive limit, extends the arms more
    #         if( arms_mean_diff_min > ARMS_ANGLE_LIMIT) | (arms_mean_diff_max < -ARMS_ANGLE_LIMIT): 
    #             print('Your arms are too low. Try to raise them slightly.')
    #         # if it is less than the negative limit, bends the arms more
    #         elif (arms_mean_diff_min < -ARMS_ANGLE_LIMIT) | (arms_mean_diff_max > ARMS_ANGLE_LIMIT):
    #             print('Your arms are too high. Try to lower them slightly.')

        
    #         ### !!! WE CAN THINK ABOUT THIS !!!

    #         # # LAST PART OF THE MOVEMENT
    #         # # Hypothetically this is while the free throw end
    #         # # Check if the arms go too much over/behind the head or not
    #         # # if it is greater than the positive limit, positioned too close to the chest or below the ideal level
    #         # if arms_max_diff > ARMS_LIMIT:
    #         #     print('Your arms are too high. Try to lower them slightly.')
    #         # # if it is less than the negative limit, extends the arms more
    #         # elif arms_max_diff < -ARMS_LIMIT:
    #         #     print('Your arms are too low. Try to raise them slightly.')

            
    #         # Check if the movements range of the rigth arm is too large respects to the GS's one
    #         if range_R_arm > range_R_arm_GS + ARMS_ANGLE_LIMIT:
    #             print('Your right arm movement is uncoordinated and too spread out. Try to maintain a smoother and more balanced motion.')

    #         # Check if the movements range of the left arm is too large respects to the GS's one
    #         if range_L_arm > range_L_arm_GS + ARMS_ANGLE_LIMIT:
    #             print('Your left arm movement is inconsistent and too spread out. Try to maintain a smoother and more balanced motion.')

    #     if legs_metric > LEGS_LIMIT:
    #         print('You should adjust a little your legs')

    #         # If the mean difference (between right and left) of the knees angles with the GS is over the acceptable limit (TOO STRAIGHT), suggest increasing the knee bend
    #         if knees_mean_diff > KNEES_ANGLE_LIMIT:
    #              print('Your posture is too straight. Try to bend your knees a little more.')
    #         # if it is below the negative limit, suggest decreasing the elbow bend
    #         elif knees_mean_diff < -KNEES_ANGLE_LIMIT:
    #             print('Your knees are too bent. Try to extend your knees a little more.')

    #     if other_metric > OTHER_LIMIT:
    #         print('You should adjust a little your back and your movement as a whole')

    #         # # TODO: check if this is correct and how is calculated the pelvis angle
    #         # if pelvis_mean_diff > PELVIS_ANGLE_LIMIT:
    #         #     print('Your posture is too rigid. Try bending forward slightly at the pelvis.')
    #         # elif pelvis_mean_diff < -PELVIS_ANGLE_LIMIT:
    #         #     print('Your posture is too bent forward. Try to straighten your pelvis a little.')

    # print(f'Your speed shooting is {speed}, try to {suggestion}')

    # -----------------------------------------------------------------------------------------


def calculate_total_percentage(final_score, metrics_limits):
    # Calculate the total possible score (sum of all 'importance' values)
    total_possible_score = sum(limit.importance for limit in metrics_limits.values())
    
    # Calculate the percentage
    percentage = round((final_score / total_possible_score) * 100)
    return percentage

def free_throw_goodness(final_score, metrics_limits):
    percentage = calculate_total_percentage(final_score, metrics_limits)

    print(f'TOTAL GOODNESS: {percentage}%.')

    if percentage <= 40:
        print("BEGINNER: You're a beginner. Focus on the basics.\n")
    elif percentage <= 60:
        print("AMATEUR PLAYER: The basics are set, now refine your technique.\n")
    elif percentage <= 80:
        print("PRO PLAYER: Solid technique! Just fine-tune your shot.\n")
    elif percentage <= 90:
        print("ELITE PLAYER: Excellent! You're almost at the top.\n")
    else:
        print("GOAT PLAYER: Unbelievable! Your form matches the Gold Standard.\n")

# Compute the performance of the shooting with respect to the gold standard
def compute_performance(vertices, vertices_compare, bones_list, len_p, len_p_compare):

    """
    Parameters:
    - vertices: the original skeleton (vertex frames)
    - vertices_compare: the compared skeleton (vertex frames)
    - bones_list: list of the bones
    - len_p: player free throw length
    - len_p_compare: gold standard free throw length
    """
    
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


    # Keeps track of the score through all the limits and metrics
    actual_score = 0
    
    # Calculate the difference in frames between the free throw of the player and the one of the GS
    speed_diff = len_p - len_p_compare


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

    ## Define some important values to compute the metric

    # Apply the coefficent to the speed differnce
    speed_diff = abs(speed_diff) * SPEED_METRIC

    # Difference in minimum right elbow angles between the Player and the Gold Standard
    elbows_R_min_diff = results_min_diff['RightElbow_diff'] 
    # Difference in minimum left elbow angles between the Player and the Gold Standard
    elbows_L_min_diff = results_min_diff['LeftElbow_diff']
    # Difference in average minimum arm angles (right and left) between the player and the gold standard
    arms_mean_diff_min = results_min_diff['Arms_mean_diff']
    # Difference in average maximum arm angles (right and left) between the player and the gold standard
    arms_mean_diff_max = results_max_diff['Arms_mean_diff']
    # Difference in average minimum knee angles (right and left) between the player and the gold standard
    knees_mean_diff = results_min_diff['Knees_mean_diff']

    # TODO: check it, because could be due to the jump
    pelvis_mean_diff_min = results_min_diff['Pelvis_mean_diff']
    # print ('pelvis mean', pelvis_mean_diff)


    # Sum the differences between the angles of the right elbow, left elbow, the min mean and max mean of the arms with the Gold Standard ones
    # The elbows have more importance in a free throw, so we keep them separately
    arms_angles_metric = sum(map(abs, (elbows_R_min_diff, elbows_L_min_diff, arms_mean_diff_min, arms_mean_diff_max)))
    # Calculate arm metrics by adding the distances and angles of the principal components of the upper body multiplied by a coefficient
    arms_metric = (np.sum(arms_distances) + arms_angles_metric) * ARMS_METRIC

    # Sum the differences between the angles of the min mean and max mean of the pelvis, and the min mean of the knees with the Gold Standard ones
    legs_angles_metric = abs(pelvis_mean_diff_min + results_max_diff['Pelvis_mean_diff'] + knees_mean_diff)
    # Calculate leg metrics by adding the distances and angles of the principal components of the lower body multiplied by a coefficient
    legs_metric = (np.sum(legs_distances) + legs_angles_metric) * LEGS_METRIC
    # Calculate the metrics of the remaining body components multiplied by a coefficient
    other_metric = np.sum(other_distances) * OTHER_METRIC
    # Calculate the overall summing all the metrics
    overall_metric = arms_metric + legs_metric + other_metric + speed_diff


    # print(f'Total arms: {arms_angles_metric}')
    # print(f'Total legs: {legs_angles_metric}')
    print(f'Arms metric: {arms_metric}')
    print(f'Legs metric: {legs_metric}')
    print(f'Other metric: {other_metric}')
    print(f'Overall metric: {overall_metric}')


    # print ('elbow_R', elbows_R_min_diff)
    # print ('elbow_L', elbows_L_min_diff)
    # arms_min_diff = results_min_diff['RightArm_diff'] + results_min_diff['LeftArm_diff']
    # arms_max_diff = results_max_diff['RightArm_diff'] + results_max_diff['LeftArm_diff']
 
    # print('arms_mean_diff_min', arms_mean_diff_min)
    # print('arms_mean_diff_max', arms_mean_diff_max)
    # range_arms_diff = arms_max_diff - arms_min_diff


    # We calculate the max range of the right and left arms during the free throw
    # Hypothetically this give us an indication on the smoothing of the shot
    range_R_arm = results_max_angles['RightArm_angles'][0] - results_min_angles['RightArm_angles'][0]
    range_L_arm = results_max_angles['LeftArm_angles'][0] - results_min_angles['LeftArm_angles'][0]
    range_R_arm_GS = results_max_angles['RightArm_angles'][1] - results_min_angles['RightArm_angles'][1]
    range_L_arm_GS = results_max_angles['LeftArm_angles'][1] - results_min_angles['LeftArm_angles'][1]



    # print (f'r: {range_R_arm}, l: {range_L_arm}')
    # print (f'r: {range_R_arm_GS}, l: {range_L_arm_GS}')

    # print('arms diff mx-min', range_arms_diff)
    # knees_min_diff = results_min_diff['RightKnee_diff'] + results_min_diff['LeftKnee_diff']
    
    
    # if overall_metric == 0.0:
    #     print(f'\n Overall score: 100%')
    # else:
    #     overall_percentage = (limits['OVERALL_LIMIT'].value / overall_metric) * 100
    #     print(f'\n Overall score: {overall_percentage:.2f}%')
    

    # # Calculate total percentage
    # percentage = calculate_total_percentage(metrics, limits)
    # print(f"Total performance percentage: {percentage:.2f}%")


    suggestions_parameters = {
        "overall_metric": overall_metric,
        "arms_metric": arms_metric,
        "elbows_R_min_diff": elbows_R_min_diff,
        "elbows_L_min_diff": elbows_L_min_diff,
        "arms_mean_diff_min": arms_mean_diff_min,
        "arms_mean_diff_max": arms_mean_diff_max,
        "range_R_arm": range_R_arm,
        "range_R_arm_GS": range_R_arm_GS,
        "range_L_arm": range_L_arm,
        "range_L_arm_GS": range_L_arm_GS,
        "legs_metric": legs_metric,
        "knees_mean_diff": knees_mean_diff,
        "other_metric": other_metric,
        "pelvis_mean_diff_min": pelvis_mean_diff_min,
        "speed_diff": speed_diff
    }


    actual_score = define_suggestions(actual_score, suggestions_parameters)

    # TODO: try a way to show the goodness before the suggestions
    free_throw_goodness(actual_score, LIMITS)
