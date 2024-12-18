import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from global_constants import *

def filterData(filePath, data):

    if data is None or filePath is None:
        print(f'Error: Invalid data or file path\n')
        return None

    fName = os.path.basename(filePath)
    fName, fExt = os.path.splitext(fName)

    if fExt == EXTENSIONS[Extension.csv]:
        # perform spline interpolation
        if not BASKET:
            splineInterpolation(data)
        TMP_VARIABLE = True
        if TMP_VARIABLE:
        # perform Kalman filter
            for model in data:
                print(f'Performing Kalman filter on model: {model.getName()}')
                kalmanFilter(model)
        
        print(f'Data correctly filtered\n')
    elif fExt == EXTENSIONS[Extension.bvh] or fExt == EXTENSIONS[Extension.c3d]:
        print(f'Info: Filtering for file extension: {fExt} is not supported\n')
    else:
        print(f'Error: Invalid file extension: {fExt}\n')

# Function to initialize a Kalman filter.
# It sets up the filter with specified parameters for time step, velocity,
# acceleration, process noise, and measurement noise.
# The function defines the transition and measurement matrices,
# as well as noise covariance matrices, and initializes the filter's state.
def initKalmanFilter(dt=5, vel=1, acc=1, processNoiseStD=1e-5, measurementNoiseStD=1e-5):
    # time step, we are assuming it constant for simplicity
    dtTo2 = 0.5 * dt**2

    KF_DINAMIC_PARAMETERS = 9
    measureParameters = 3

    # instantiate and initialize the Kalman filter
    kalman = cv2.KalmanFilter(KF_DINAMIC_PARAMETERS, measureParameters)

    kalman.measurementMatrix = np.eye(measureParameters, KF_DINAMIC_PARAMETERS, dtype=np.float32)

    kalman.transitionMatrix = np.array([
        [1, 0, 0, vel*dt, 0, 0, acc*dtTo2, 0, 0],
        [0, 1, 0, 0, vel*dt, 0, 0, acc*dtTo2, 0],
        [0, 0, 1, 0, 0, vel*dt, 0, 0, acc*dtTo2],
        [0, 0, 0, 1, 0, 0, vel*dt, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, vel*dt, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, vel*dt],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]], np.float32)

    kalman.processNoiseCov = np.eye(KF_DINAMIC_PARAMETERS, dtype=np.float32) * processNoiseStD

    kalman.measurementNoiseCov = np.eye(measureParameters, dtype=np.float32) * measurementNoiseStD

    # Initialize state
    kalman.statePost = np.zeros((KF_DINAMIC_PARAMETERS, 1), dtype=np.float32)
    kalman.statePre = np.zeros((KF_DINAMIC_PARAMETERS, 1), dtype=np.float32)

    return kalman

# Function to estimate missing values in a dataset using a Kalman filter.
# It processes 3D positional data (X, Y, Z) from a source model,
# filling in missing values where coordinates are zero.
def kalmanFilter(sourceModel):

    if BASKET:
        time = np.linspace(0, 40, len(sourceModel.positions[X]))
    else:
        time = sourceModel.time
    # input data
    data = {
        TIME_SHORT: time,
        X: sourceModel.positions[X],
        Y: sourceModel.positions[Y],
        Z: sourceModel.positions[Z]
    }

    # convert to pandas DataFrame
    df = pd.DataFrame(data)

    # time step, we are assuming it constant for simplicity
    dt = df[TIME_SHORT].diff().mean()
    kalman = initKalmanFilter(dt)

    # prepare storage for estimated results
    estimatedCoords = np.zeros((len(df), 3), dtype=np.float32)

    # detetct missing values and apply Kalman filter
    for i, row in df.iterrows():
        if (row[X] == 0.0 or row[X] == None or type(row[X]) != np.float64) and (row[Y] == 0.0 or row[Y] == None or type(row[Y]) != np.float64) and (row[Z] == 0.0 or row[Z] == None or type(row[Z]) != np.float64):
            kalman.predict()
            print(f'Missing values detected: {row[X]}, {row[Y]}, {row[Z]}')
        else:
            measurement = np.array([row[X], row[Y], row[Z]], dtype=np.float32)
            kalman.predict()
            kalman.correct(measurement)

        estimatedCoords[i] = measurement.flatten() #kalman.statePost[:3].flatten()

    # fill in the missing coordinates in the DataFrame
    df_estimated = df.copy()
    df_estimated[[X, Y, Z]] = estimatedCoords

    # copy the data back to the original model
    sourceModel.kfPositions[X] = df_estimated[X].to_list()
    sourceModel.kfPositions[Y] = df_estimated[Y].to_list()
    sourceModel.kfPositions[Z] = df_estimated[Z].to_list()

    if PLOT_CHART:
        # plot the results
        plotData("Kalman Filter", sourceModel.positions, sourceModel.kfPositions)

# Function to perform cubic spline interpolation to estimate and fill in missing
# 3D positional data (X, Y, Z) for a list of models (markers).
# It identifies missing entries, interpolates the data based on available time points,
# and fills in the missing values.
def splineInterpolation(data):

    print(f'Performing spline interpolation\n')

    markerList = []
    # obtaining a list of list of positions in time
    for model in data:
        tempList = list(zip(model.positions[X], model.positions[Y], model.positions[Z]))
        markerList.append(tempList)
    markerList = np.array(markerList)

    # number of markers
    num_markers, num_frames, _ = markerList.shape

    # identify time entries with missing values
    missing_indices = np.where(
        ((markerList == (0, 0, 0)).all(axis=0).all(axis=1)) |  # (0, 0, 0) check
        ((markerList == None).all(axis=0).all(axis=1)))
    # available_indices = np.where((~(markerList == (0, 0, 0)).all(axis=0).all(axis=1)) 
                                # | ((np.isfinite(markerList).all(axis=2))))[0]
    available_indices = np.setdiff1d(np.arange(num_frames), missing_indices)
    # available_indices = np.sort(available_indices)
    

    # function to interpolate and fill missing values for each marker
    def interpolate_and_fill(markerList, missing_indices, available_indices):
        filledMarkerList = np.copy(markerList)
        t = available_indices  # Time points for available data

        for marker in range(num_markers):
            for i in range(3):  # For x, y, z
                values = markerList[marker, available_indices, i]
                cs = CubicSpline(t, values)
                filled_values = cs(missing_indices)
                filledMarkerList[marker, missing_indices, i] = filled_values

        return filledMarkerList

    # fill the missing data
    filledMarkerList = interpolate_and_fill(markerList, missing_indices, available_indices)
    for i, model in enumerate(data):
            xList, yList, zList = zip(*filledMarkerList[i])
            model.splPositions[X] = xList
            model.splPositions[Y] = yList
            model.splPositions[Z] = zList

    print(f'Data correctly filtered\n')

    if PLOT_CHART:
        # plot the results for each marker
        for i, model in enumerate(data):
            plotData("Spline interpolation for marker %d" % i, model.positions, model.splPositions)

# Function to visualize 3D positional data by plotting it on a 3D scatter plot.
# It takes the original data points and optionally a second set of data points
# (after some operation) # and displays them in two separate charts side by side.
# # The first chart shows the original data, while the second chart (if provided)
# shows the processed data, labeled according to the specified operation.
def plotData(operation, firstDataPoints, secondDataPoints=None):

    # extract data
    X1 = np.array(firstDataPoints[X])
    Y1 = np.array(firstDataPoints[Y])
    Z1 = np.array(firstDataPoints[Z])

    if secondDataPoints is not None:
        X2 = np.array(secondDataPoints[X])
        Y2 = np.array(secondDataPoints[Y])
        Z2 = np.array(secondDataPoints[Z])

    # plot the results in 3D on two separate charts
    fig = plt.figure(figsize=(14, 6))

    # plot for original data
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(X1, Y1, Z1, c='r', label='Original Data')
    ax1.set_xlabel(X)
    ax1.set_ylabel(Y)
    ax1.set_zlabel(Z)
    ax1.set_title('Original Data')

    if secondDataPoints is not None:
        # plot for new data
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.scatter(X2, Y2, Z2, c='b', label=f'{operation} Data')
        ax2.set_xlabel(X)
        ax2.set_ylabel(Y)
        ax2.set_zlabel(Z)
        ax2.set_title(f'{operation} Data')

    plt.tight_layout()
    plt.show()
