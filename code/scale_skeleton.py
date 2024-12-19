import numpy as np
# import open3d as o3d
# import cv2
# import os
# from global_constants import *
# from I_O import *
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# Compute the joint length with the euclidean distance between 2 joints.
def compute_joint_length(point_a, point_b):
    return np.linalg.norm(point_b - point_a)

# resize the skeleton scaling each joint
def resize_skeleton(skeleton, target_joints_length, current_joints_length):
    scale_ratio = target_joints_length / current_joints_length
    scaled_skeleton = skeleton * scale_ratio
    return scaled_skeleton

def resize_skeleton_partial(skeleton, target_joints_length, current_joints_length, joints_to_scale):
    """
    Resize part of the skeleton by scaling specified joints.

    Args:
        skeleton (list of tuples): The 3D coordinates of the skeleton [(x1, y1, z1), (x2, y2, z2), ...].
        target_joints_length (float): The desired target length for the selected joints.
        current_joints_length (float): The current length of the selected joints.
        joints_to_scale (list of int): Indices of the joints to scale.

    Returns:
        list of tuples: The resized skeleton with scaled joints.
    """
    # Calculate the scale ratio
    scale_ratio = target_joints_length / current_joints_length

    # Scale only the specified joints
    scaled_skeleton = skeleton[:]
    for joint_index in joints_to_scale:
        x, y, z = scaled_skeleton[joint_index]
        scaled_skeleton[joint_index] = (x * scale_ratio, y * scale_ratio, z * scale_ratio)

    return scaled_skeleton

def scale_single_frame(lines, frame, vertices):
    target_joints_length = 10

    # List of joints at the single frame
    skeleton = vertices[frame]

    joints_length = 0
    for indices in lines:
        index_1, index_2 = indices
        joints_length += compute_joint_length(skeleton[index_1], skeleton[index_2])

    skeleton = resize_skeleton_partial(skeleton, target_joints_length, joints_length, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    return skeleton
    
# def skeletonJointsPlot(data, fName):
  
#   jointsGraph = {
#     'Hips' : ['Spine',  'LeftUpLeg', 'RightUpLeg'],
#     'Spine' : ['Spine1'],
#     'Spine1' : ['Spine2'],
#     'Spine2' : ['Neck'],
#     'Neck' : ['Head',  'LeftShoulder', 'RightShoulder'],
#     #'Head' : ['Neck'], #we know  head is not connected to anything new
#     'LeftShoulder' : ['LeftArm'],
#     'LeftArm' : ['LeftForeArm', 'LeftForeArmRoll'],
#     'LeftForeArm' : ['LeftHand'],
#     'LeftForeArmRoll' : ['LeftHand'],
#     #'LHand' : ['LFArm'], #we know  hand is not connected to anything new
#     'RightShoulder' : ['RightArm'],
#     'RightArm' : ['RightForeArm', 'RightForeArmRoll'],
#     'RightForeArm' : ['RightHand'],
#     'RightForeArmRoll' : ['RightHand'],
#     #'RightHand' : ['RightForeArmRoll'], #we know  hand is not connected to anything new
#     'LeftUpLeg' : ['LeftLeg'],
#     'LeftLeg' : ['LeftFoot'],
#     'LeftFoot' : ['LeftToeBase'],
#     #'LeftToeBase' : ['LeftFoot'], #we know  toe is not connected to anything new
#     'RightUpLeg' : ['RightLeg'],
#     'RightLeg' : ['RightFoot'],
#     'RightFoot' : ['RightToeBase'],
#     #'RToe' : ['RFoot'] #we know  toe is not connected to anything new
#   }

#   for model in data:
#     if BASKET or model._description['Type'] != 'Marker':
#       points = list(zip(model._positions['x'], model._positions['y'], model._positions['z']))
#       bonesPosDict[model._description['Name']] = points

#   # get the lists of points from the dictionary values
#   lists_of_points = list(bonesPosDict.values())

#   # use zip to combine corresponding points from each list into tuples
#   lists_of_points = list(zip(*lists_of_points))
  
#   vertices = [] # numpy array
#   # we adapt a list of tuples [xyz, xyz, xyz, xyz] to Open3D
#   for skeletonPoints in lists_of_points:
#     vertices.append(np.array(skeletonPoints))


#   # create a Visualizer object
#   visualizer = setVisualizer(10.0)

#   # create line set to represent edges in the graph
#   lines = []
#   for start, ends in jointsGraph.items():
#       start_idx = getIndex(start, bonesPosDict)
#       for end in ends:
#           end_idx = getIndex(end, bonesPosDict)
#           lines.append([start_idx, end_idx])

#   skeleton = scale_single_frame(lines, 0, vertices)

#   vertices = skeleton

#   line_set = o3d.geometry.LineSet()
#   # line_set.points = o3d.utility.Vector3dVector(vertices)
#   line_set.lines = o3d.utility.Vector2iVector(lines)

#   # set line color (e.g., red)
#   line_color = [1, 0, 0] # RGB color (red in live and blue in saved video)

#   # create a LineSet with colored lines
#   line_set.colors = o3d.utility.Vector3dVector(np.tile(line_color, (len(lines), 1)))

#   while True:
#     print("Do you want to save the video?")
#     print("1. YES")
#     print("0. NO")

#     option = input("Enter your choice: ")
#     if option == '1':
#       print("Saving and showing video...")

#       if not checkDir(mode=SAVE, path=SAVE_VIDEO_PATH):
#         print(f"Error: Impossible to save the video in the directory: {SAVE_VIDEO_PATH}")
#         return

#       # Iteration over all the point clouds
#       videoWriter = cv2.VideoWriter(os.path.join(SAVE_VIDEO_PATH, (fName + '.avi')), cv2.VideoWriter_fourcc(*'DIVX'), FPS, (720, 480))
#       for i, skeletonPoints in enumerate(vertices):
#         if i%3 == 0:
#           continue
#         if i%4 == 0:
#           continue
#         visualizer.clear_geometries()
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(skeletonPoints)
#         line_set.points = o3d.utility.Vector3dVector(skeletonPoints)
#         visualizer.add_geometry(pcd)
#         visualizer.add_geometry(line_set)
#         visualizer.update_geometry(pcd)
#         visualizer.update_geometry(line_set)
#         visualizer.poll_events()
#         visualizer.update_renderer()
#         image = visualizer.capture_screen_float_buffer(do_render=True)
#         image = np.asarray(image)
#         image = (image * 255).astype(np.uint8)  # Convert to 8-bit image
#         # Write the frame to the video
#         videoWriter.write(image)
#       videoWriter.release()
#       visualizer.destroy_window()
#       print("Video saved")
#       break

#     elif option == '0':
#       print("Showing video...")
#       for i, skeletonPoints in enumerate(vertices):
#         if i%3 == 0:
#           continue
#         if i%4 == 0:
#           continue
#         visualizer.clear_geometries()
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(skeletonPoints)
#         line_set.points = o3d.utility.Vector3dVector(skeletonPoints)
#         visualizer.add_geometry(pcd)
#         visualizer.add_geometry(line_set)
#         visualizer.update_geometry(pcd)
#         visualizer.update_geometry(line_set)
#         visualizer.poll_events()
#         visualizer.update_renderer()
      
#       visualizer.destroy_window()
#       print("Video not saved")
#       break

#     else:
#       print("Invalid input, try again.")
    