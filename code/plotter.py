import open3d as o3d
import cv2
import os
from global_constants import *
from I_O import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scale_skeleton import *

# The number of frames per second for the videos we want to save
FPS = 90

# An interactive menu to select what to plot
def plotData(filePath=None, data=None, compareFilePath=None, compareData=None, COMPARE = False):

    if data is None or filePath is None:
      print(f'Error: Invalid data or file path\n')

    fName = os.path.basename(filePath)
    fNameCompare = os.path.basename(compareFilePath) if COMPARE else None
    fName, fExt = os.path.splitext(fName)
    
    if fExt == EXTENSIONS[Extension.csv]:
      # if fName == RIGID_BODY:
      #   while True:
      #     print(f'Please select the type of rigid body visualization')
      #     print(f'1. Original rigid body points')
      #     print(f'2. Kalman filtered rigid body points')
      #     print(f'3. Spline interpolation estimated rigid body points')
      #     print(f'0. Exit the program')

      #     option = input(f'Enter your choice: ')
      #     if option == '1':
      #       print(f'Original rigid body points')
      #       markerRigidBodyPlot(data, fName, typeOfFiltering=None)
      #       return True

      #     elif option == '2':
      #       print(f'Kalman filtered rigid body points')
      #       markerRigidBodyPlot(data, fName, KALMAN_FILTER)
      #       return True

      #     elif option == '3':
      #       print(f'Spline interpolation estimated rigid body points')
      #       markerRigidBodyPlot(data, fName, SPLINE_INTERPOLATION)
      #       return True

      #     elif option == '0':
      #       # exit the loop and end the program
      #       break

      #     else:
      #         print(f'Invalid input, try again.')

      #     print()

      #   return True
      # elif COMPARE:
      if COMPARE:
        print(f'Plotting superposed skeletons of {fName} and {fNameCompare}')
        skeletonJointsPlot(data, fName, compareData, fNameCompare)
        return True
      else:
        while True:
            print(f'Please select the type of skeleton visualization:')
            print(f'1. Skeleton with markers')
            print(f'2. Skeleton joints')
            print(f'0. Exit the program')

            option = input(f'Enter your choice: ')
            if option == '1':
              print(f'Skeleton with markers')
              skeletonMarkerPlot(data, fName)
              return True

            elif option == '2':
              print(f'Skeleton joints')
              skeletonJointsPlot(data, fName)
              return True

            elif option == '0':
              # exit the loop and end the program
              break

            else:
                print(f'Invalid input, try again.')
        
        print()

        return True

    elif fExt == EXTENSIONS[Extension.bvh] or fExt == EXTENSIONS[Extension.c3d]:
        print(f'Info: Plotting for file extension: {fExt} is not supported\n')
    else:
        print(f'Error: Invalid file extension: {fExt}\n')

def getIndex(start, dictionary):
  for key in dictionary.keys():
    if start in key:
      return list(dictionary.keys()).index(key)

  return -1

# Function to render and optionally saves a sequence of 3D marker frames using a visualizer.
# It processes each frame by updating the visualizer with new point cloud data and either
# captures and writes the frames to a video file or simply displays them.
# The function skips certain frames to reduce computation and finally releases resources
# or closes the visualizer window.
def visualizeSequence(visualizer, markersList, fName, typeOfFiltering, SAVE_VIDEO):

  videoWriter = None
  if SAVE_VIDEO:
    
    if not checkDir(mode=SAVE, path=SAVE_VIDEO_PATH):
      print(f"Error: Impossible to save the video in the directory: {SAVE_VIDEO_PATH}")
      return

    if typeOfFiltering is not None:
      fileName = os.path.join(SAVE_VIDEO_PATH, (typeOfFiltering + fName + '.avi'))
    else:
      fileName = os.path.join(SAVE_VIDEO_PATH, (fName + '.avi'))

    videoWriter = cv2.VideoWriter(fileName, cv2.VideoWriter_fourcc(*'DIVX'), FPS, (720, 480))
    for i, marker in enumerate(markersList):
      if i % 3 == 0: # skip every 3rd frame to reduce computations
        continue
      if i % 4 == 0: #skip every 4th frame to reduce computations
        continue
      visualizer.clear_geometries() # clears any existing geometries from the visualizer
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(marker) # sets the points of the pcd from the points in the marker tuple
      visualizer.add_geometry(pcd) # adds the pcd to the visualizer for rendering
      visualizer.update_geometry(pcd)
      visualizer.poll_events()
      visualizer.update_renderer()
      image = visualizer.capture_screen_float_buffer(do_render=True)
      image = np.asarray(image)
      image = (image * 255).astype(np.uint8)  # convert to 8-bit image
      # write the frame to the video
      videoWriter.write(image)

  else:
    for i, marker in enumerate(markersList):
      if i % 2 == 0 or i % 5 == 0: #skip every 2nd and fifth frame to reduce computations
        continue
      visualizer.clear_geometries() # clears any existing geometries from the visualizer
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(marker) #sets the points of the pcd from the points in the marker tuple
      visualizer.add_geometry(pcd) # adds the pcd to the visualizer for rendering
      visualizer.update_geometry(pcd)
      visualizer.poll_events()
      visualizer.update_renderer()

  visualizer.destroy_window()
  
  if videoWriter is not None:
    videoWriter.release()


def visualizeWithOther(allMarkers, fName, typeOfFiltering, SAVE_VIDEO):
  # Example data: points in space (x, y, z) over time (t)
  # allMarkers is a list of frames. For each frame we have a list of bones. For each bone we 
  # have a tuple x,y,z

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.set_xlim(-2000, 2000)
  ax.set_ylim(-2000, 2000)
  ax.set_zlim(2000, 2000)

  def update(frame):
      ax.cla()  # Clear the axis
      scList = []
      for bone in allMarkers[frame]:
        sc = ax.scatter(bone[0], bone[1], bone[2], c='r')
        scList.append(sc)
      ax.set_xlim(-2000, 2000)
      ax.set_ylim(-2000, 2000)
      ax.set_zlim(-2000, 2000)
      ax.view_init(elev=20, azim=frame)  # Rotate view
      return scList

  ani = FuncAnimation(fig, update, frames=len(allMarkers), interval=10)

  # To display
  plt.show()


def setVisualizer(pointSize):

  # creation of the visualizer window
  visualizer = o3d.visualization.Visualizer()
  visualizer.create_window(window_name='Open3D', width=720, height=480)
  visualizer.get_render_option().background_color = np.asarray([0, 0, 0]) #black background
  visualizer.get_render_option().point_size = pointSize
  return visualizer

# Function to visualize 3D marker data from a list of models. Depending on whether filtering
# is applied, it either uses original or filtered marker positions. The function transforms
# the marker data into a format suitable for visualization,
# then uses a visualizer to display a sequence of the marker points.
def markerRigidBodyPlot(data, fName, typeOfFiltering):

  allMarkers = []
  # plot the original points
  if typeOfFiltering == None:
    # transform from dictionary[(X:x1,x2...), (Y:y1,y2...), (Z:z1,z2...)] to list of tuples (list[x, y, z])
    for model in data:
      points = list(zip(model._positions[X], model._positions[Y], model._positions[Z]))
      allMarkers.append(points)

  else: # takes the filtered points based on the filter requested
    # transform from dictionary[(X:x1,x2...), (Y:y1,y2...), (Z:z1,z2...)] to list of tuples (list[x, y, z])
    for model in data:
      filteredPoints = getattr(model, typeOfFiltering)
      points = list(zip(filteredPoints[X],filteredPoints[Y],filteredPoints[Z]))
      allMarkers.append(points)
  
  # transform to list of tuples [xyz, xyz, xyz, xyz] where each tuple is a point cloud
  allMarkers = list(zip(*allMarkers)) 
  while True:
    print("Do you want to save the video?")
    print("1. YES")
    print("0. NO")

    option = input("Enter your choice: ")
    if option == '1':
      print("Saving and showing video...")
      SAVE_VIDEO = True
      visualizer = setVisualizer(10.0)
      visualizeSequence(visualizer, allMarkers, fName, typeOfFiltering, SAVE_VIDEO)
      # visualizeWithOther(allMarkers, fName, typeOfFiltering, SAVE_VIDEO)
      print("Video saved")
      break

    elif option == '0':
      print("Showing video...")
      SAVE_VIDEO = False
      visualizer = setVisualizer(10.0)
      visualizeSequence(visualizer, allMarkers, fName, typeOfFiltering, SAVE_VIDEO)
      # visualizeWithOther(allMarkers, fName, typeOfFiltering, SAVE_VIDEO)
      print("Video not saved")
      break

    else:
      print("Invalid input, try again.")

# saves the video or just displays it using the `visualizeSequence` function.
def skeletonMarkerPlot(data, fName):

  bonesMarkerList = []
  for model in data:
    if BASKET or model._description['Type'] == 'Marker':
      points = list(zip(model._positions['x'], model._positions['y'], model._positions['z']))
      bonesMarkerList.append(points)

  bonesMarkerList = list(zip(*bonesMarkerList)) #list of tuples [xyz, xyz, xyz, xyz]

  while True:
    print("Do you want to save the video?")
    print("1. YES")
    print("0. NO")

    option = input("Enter your choice: ")
    if option == '1':
      print("Saving and showing video...")
      SAVE_VIDEO = True
      visualizer = setVisualizer(8.0)
      visualizeSequence(visualizer, bonesMarkerList, fName, None, SAVE_VIDEO)
      # visualizeWithOther(bonesMarkerList, fName, None, SAVE_VIDEO)
      print("Video saved")
      break

    elif option == '0':
      print("Showing video...")
      SAVE_VIDEO = False
      visualizer = setVisualizer(8.0)
      visualizeSequence(visualizer, bonesMarkerList, fName, None, SAVE_VIDEO)
      # visualizeWithOther(bonesMarkerList, fName, None, SAVE_VIDEO)
      print("Video not saved")
      break

    else:
      print("Invalid input, try again.")

# Function to visualize 3D skeleton data by plotting the joints and connecting bones.
# It constructs a graph representing the skeleton structure, where each joint is connected
# to others according to a predefined hierarchy.
def skeletonJointsPlot(data, fName, compareData=None, fNameCompare=None):

  bonesPosDict = {}
  bonesPosDictCompare = {}
  
  # this is the skeleton graph for optitrack 
  # jointsGraph = {
  #   'Hip' : ['Ab', 'RThigh', 'LThigh'],
  #   'Ab' : ['Chest'],
  #   'Chest' : ['Neck'],
  #   'Neck' : ['Head', 'RShoulder', 'LShoulder'],
  #   #'Head' : ['Neck'], #we know  head is not connected to anything new
  #   'LShoulder' : ['LUArm'],
  #   'LUArm' : ['LFArm'],
  #   'LFArm' : ['LHand'],
  #   #'LHand' : ['LFArm'], #we know  hand is not connected to anything new
  #   'RShoulder' : ['RUArm'],
  #   'RUArm' : ['RFArm'],
  #   'RFArm' : ['RHand'],
  #   #'RHand' : ['RFArm'], #we know  hand is not connected to anything new
  #   'LThigh' : ['LShin'],
  #   'LShin' : ['LFoot'],
  #   'LFoot' : ['LToe'],
  #   #'LToe' : ['LFoot'], #we know  toe is not connected to anything new
  #   'RThigh' : ['RShin'],
  #   'RShin' : ['RFoot'],
  #   'RFoot' : ['RToe'],
  #   #'RToe' : ['RFoot'] #we know  toe is not connected to anything new
  # }
  
  jointsGraph = {
    'Hips' : ['Spine',  'LeftUpLeg', 'RightUpLeg'],
    'Spine' : ['Spine1'],
    'Spine1' : ['Spine2', 'LeftShoulder', 'RightShoulder'],
    'Spine2' : ['Neck'],
    'Neck' : ['Head',  'LeftShoulder', 'RightShoulder'],
    #'Head' : ['Neck'], #we know  head is not connected to anything new
    'LeftShoulder' : ['LeftArm', 'RightShoulder', 'Spine1', 'Spine2'],
    'LeftArm' : ['LeftForeArm', 'Spine'],
    'LeftForeArm' : ['LeftForeArmRoll'],
    'LeftForeArmRoll' : ['LeftHand'],
    #'LHand' : ['LFArm'], #we know  hand is not connected to anything new
    'RightShoulder' : ['RightArm', 'LeftShoulder', 'Spine1', 'Spine2'],
    'RightArm' : ['RightForeArm', 'Spine'],
    'RightForeArm' : ['RightForeArmRoll'],
    'RightForeArmRoll' : ['RightHand'],
    #'RightHand' : ['RightForeArmRoll'], #we know  hand is not connected to anything new
    'LeftUpLeg' : ['LeftLeg'],
    'LeftLeg' : ['LeftFoot'],
    'LeftFoot' : ['LeftToeBase'],
    #'LeftToeBase' : ['LeftFoot'], #we know  toe is not connected to anything new
    'RightUpLeg' : ['RightLeg'],
    'RightLeg' : ['RightFoot'],
    'RightFoot' : ['RightToeBase'],
    #'RToe' : ['RFoot'] #we know  toe is not connected to anything new
  }

  for model in data:
    if BASKET or model._description['Type'] != 'Marker':
      points = list(zip(model._positions['x'], model._positions['y'], model._positions['z']))
      bonesPosDict[model._description['Name']] = points
  
  if compareData:
    for model in compareData:
      if BASKET or model._description['Type'] != 'Marker':
        points = list(zip(model._positions['x'], model._positions['y'], model._positions['z']))
        bonesPosDictCompare[model._description['Name']] = points
  
  # get the lists of points from the dictionary values
  lists_of_points = list(bonesPosDict.values())
  lists_of_pointsCompare = list(bonesPosDictCompare.values()) if compareData else None

  # use zip to combine corresponding points from each list into tuples
  lists_of_points = list(zip(*lists_of_points))
  if compareData:
    lists_of_pointsCompare = list(zip(*lists_of_pointsCompare))
  
  if compareData:
      # manually cut the lists of points to select only the shoots
      # lists_of_points = lists_of_points[2520:2642] #edo
      # lists_of_pointsCompare = lists_of_pointsCompare[865:1030] #nick
      lists_of_points, lists_of_pointsCompare = downsample_video(lists_of_points, lists_of_pointsCompare)
  
  vertices = [] # numpy array
  verticesCompare = []
  # we adapt a list of tuples [xyz, xyz, xyz, xyz] to Open3D
  for skeletonPoints in lists_of_points:
    vertices.append(np.array(skeletonPoints))
  if compareData:
    for skeletonPoints in lists_of_pointsCompare:
      verticesCompare.append(np.array(skeletonPoints))

  # create a Visualizer object
  visualizer = setVisualizer(10.0)

  # create line set to represent edges in the graph
  lines = []
  for start, ends in jointsGraph.items():
      start_idx = getIndex(start, bonesPosDict)
      for end in ends:
          end_idx = getIndex(end, bonesPosDict)
          lines.append([start_idx, end_idx])
  lines_compare = []
  if compareData:
    for start, ends in jointsGraph.items():
        start_idx = getIndex(start, bonesPosDictCompare)
        for end in ends:
            end_idx = getIndex(end, bonesPosDictCompare)
            lines_compare.append([start_idx, end_idx])
  
  ## Scaling skeletons
  # Scale the skeletons only if the user wants to compare 2 performances
  if compareData:
    # Scale the first skeleton  
    vertices = scale_multiple_frames(lines, vertices)
    # Scale the second skeleton to compare
    verticesCompare = scale_multiple_frames(lines, verticesCompare)
    # Compute the performance of the two skeletons
    compute_performance(vertices, verticesCompare, list(bonesPosDict.keys()))

    verticesCompare = align_pelvises(vertices, verticesCompare)
  
  line_set = o3d.geometry.LineSet()
  line_set_compare = o3d.geometry.LineSet() if compareData else None
  # line_set.points = o3d.utility.Vector3dVector(vertices)
  line_set.lines = o3d.utility.Vector2iVector(lines)
  if compareData:
    line_set_compare.lines = o3d.utility.Vector2iVector(lines)
  
  # set line color (e.g., red)
  line_color = [1, 0, 0] # RGB color (red in live and blue in saved video)
  line_color_compare = [0, 0, 1] # RGB color (blue in live and red in saved video)

  # create a LineSet with colored lines
  line_set.colors = o3d.utility.Vector3dVector(np.tile(line_color, (len(lines), 1)))
  if compareData:
    line_set_compare.colors = o3d.utility.Vector3dVector(np.tile(line_color_compare, (len(lines_compare), 1)))
    
  while True:
    print("Do you want to save the video?")
    print("1. YES")
    print("0. NO")

    option = input("Enter your choice: ")
    if option == '1':
      print("Saving and showing video...")

      if not checkDir(mode=SAVE, path=SAVE_VIDEO_PATH):
        print(f"Error: Impossible to save the video in the directory: {SAVE_VIDEO_PATH}")
        return

      # Iteration over all the point clouds
      videoWriter = cv2.VideoWriter(os.path.join(SAVE_VIDEO_PATH, (fName + '.avi')), cv2.VideoWriter_fourcc(*'DIVX'), FPS, (720, 480))
      for i, skeletonPoints in enumerate(vertices):
        # if i%3 == 0:
        #   continue
        # if i%4 == 0:
        #   continue
        visualizer.clear_geometries()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(skeletonPoints)
        line_set.points = o3d.utility.Vector3dVector(skeletonPoints)
        visualizer.add_geometry(pcd)
        visualizer.add_geometry(line_set)
        visualizer.update_geometry(pcd)
        visualizer.update_geometry(line_set)
        visualizer.poll_events()
        visualizer.update_renderer()
        image = visualizer.capture_screen_float_buffer(do_render=True)
        image = np.asarray(image)
        image = (image * 255).astype(np.uint8)  # Convert to 8-bit image
        # Write the frame to the video
        videoWriter.write(image)
      
      videoWriter.release()
      visualizer.destroy_window()
      print("Video saved")
      break

    
    elif option == '0': # not saving the video
      # edo
      # starting_point = 2520
      # stopping_point = 2640
      #nick
      # starting_point = 865
      # stopping_point = 1030

      print("Showing video...")
      for i, skeletonPoints in enumerate(vertices):
        # if i%3 == 0:
        #   continue
        # if i%4 == 0:
        #   continue
        # print(f"Frame {i}")
        visualizer.clear_geometries()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(skeletonPoints)
        line_set.points = o3d.utility.Vector3dVector(skeletonPoints)
        visualizer.add_geometry(pcd)
        visualizer.add_geometry(line_set)
        visualizer.update_geometry(pcd)
        visualizer.update_geometry(line_set)
        if compareData:
          pcdCompare = o3d.geometry.PointCloud()
          pcdCompare.points = o3d.utility.Vector3dVector(verticesCompare[i])
          line_set_compare.points = o3d.utility.Vector3dVector(verticesCompare[i])
          visualizer.add_geometry(pcdCompare)
          visualizer.add_geometry(line_set_compare)
          visualizer.update_geometry(pcdCompare)
          visualizer.update_geometry(line_set_compare)
        view_control = visualizer.get_view_control()
        view_control.set_lookat([0, 0, 0]) #Sets the focus point of the camera (the point the camera looks at)
        view_control.set_front([1, 1, 1]) #Change the camera direction: 
                                            # x is only 1 (we see the right side) or -1 (we see the left side) 
                                            # y is rotation top/bottom view, the closest to 0, the more the bottom view
        view_control.set_up([-1, 1, 1]) #Sets the up direction for the camera
        view_control.set_zoom(1.4) #Sets the zoom factor of the camera (little zoom in, big zoom out)

        visualizer.poll_events()
        visualizer.update_renderer()
      
      visualizer.destroy_window()
      print("Video not saved")
      break

    else:
      print("Invalid input, try again.")