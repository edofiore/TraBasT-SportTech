import os
import sys
import cv2
import copy
import platform
import numpy as np

from enum import IntEnum

from utils import *

# This file defines key constants and settings for a software project,
# focusing on file handling, data processing, and experiment configurations.
# It includes file extensions, predefined paths, data structure definitions,
# filtering methods, I/O operations, and logging levels, centralizing these
# elements for consistent use throughout the project.

# FILE EXTENSIONS
class Extension(IntEnum):
    none = 0
    mat = 1
    csv = 2
    png = 3
    txt = 4
    jpg = 5
    pt = 6
    json = 7
    pkl = 8
    zip = 9
    bvh = 10
    c3d = 11

EXTENSIONS = (
    '',
    '.mat',
    '.csv',
    '.png',
    '.txt',
    '.jpg',
    '.pt',
    '.json',
    '.pkl',
    '.zip',
    '.bvh',
    '.c3d'
    )

# allowed extenSions for input files
INPUT_EXTENSIONS = (
    '.csv',
    )


# Get the absolute path of the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to your data folder
DATA_PATH = os.path.join(script_dir, 'Data')

# Normalize the path (removes redundant slashes like //)
DATA_PATH = os.path.normpath(DATA_PATH)

# List all files in the folder
CSV_LIST = os.listdir(DATA_PATH)

# # FILES NAMES
# SQUAT1 = 'Carlo_1'
# SQUAT2 = 'Edoardo_1'
# SQUAT3 = 'Sebastiano_Squat1_100'
# SQUAT4 = 'Nick_1'
# SQUAT5 = 'Edoardo_2'
# SQUAT6 = 'Carlo_2-cut'
# SQUAT7 = 'Edoardo_1-cut'
# SQUAT8 = 'Nick_1-cut'
# SQUAT9 = 'Edoardo_1-bad-cut_1'
# SQUAT10 = 'Edoardo_1-bad-cut_2'
# RIGID_BODY = 'rigid_body'

ANIMATION = 'animation'
MARKER = 'markers'

# CSV_LIST = [SQUAT1 + EXTENSIONS[Extension.csv], SQUAT2 + EXTENSIONS[Extension.csv], 
#             SQUAT3 + EXTENSIONS[Extension.csv], RIGID_BODY + EXTENSIONS[Extension.csv], 
#             SQUAT4 + EXTENSIONS[Extension.csv], SQUAT5 + EXTENSIONS[Extension.csv],
#             SQUAT6 + EXTENSIONS[Extension.csv], SQUAT7 + EXTENSIONS[Extension.csv],
#             SQUAT8 + EXTENSIONS[Extension.csv], SQUAT9 + EXTENSIONS[Extension.csv],
#             SQUAT10 + EXTENSIONS[Extension.csv]]

OTHER_LIST = [ANIMATION + EXTENSIONS[Extension.bvh], MARKER + EXTENSIONS[Extension.c3d]]
FPS_LIST = ['60fps', '360fps']

#TYPE OF FILTERING
KALMAN_FILTER = 'kfPositions'
SPLINE_INTERPOLATION = 'splPositions'

# PATHS
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, f'Data')
SAVE_PATH = os.path.join(ROOT_PATH, f'Saved')
SAVE_VIDEO_PATH = os.path.join(ROOT_PATH, f'Video')
ACTOR_DATA_PATH = os.path.join(ROOT_PATH, f'Results', f'Task_3', f'UE5', f'actor_data.json')
FRAME_IMAGE_PATH = os.path.join(ROOT_PATH, f'Results', f'Task_3', f'UE5', f'CVMap.')
CAMERA_DATA_PATH = os.path.join(ROOT_PATH, f'Results', f'Task_3', f'UE5', f'camera_data.json')

# # used for testing purposes
# SRC_FILE = SQUAT1 + EXTENSIONS[Extension.csv]
SRC_PATH = os.path.join(DATA_PATH, CSV_LIST[0])

# CSV STRUCTURE
# the application is thought to work with CVS data version 1.23
# any other version is not supported
CSV_VERSION = '1.24'

CSV_HEADER_FILE_ROW = 0 # file header row start number
CSV_HEADER_FILE_LEN = 1 # file header dimension
CSV_VERSION_COLUMN = 1 # file header column number for version info
CSV_HEADER_DATA_ROW = 2 # data header row start number
CSV_HEADER_DATA_LEN = 5 # data header dimension
CSV_DATA_COLUMN = 2 # data column start number

IGNORE_DATA = ['Marker'] # columns to be ignored

# BVH STRUCTURE
BHV_DIR = "bvh_reader"
ANIMATION = 'animation'
NAMES = 'names'

# C3D STRUCTURE
C3D_ANALOG = 'analog'
C3D_SCALE_FACTOR = 'scale_factor'
C3D_ERR_EST = 'err_est' # estimated error
C3D_CAMERA_NR = 'camera' # number of cameras that registered the point
C3D_POINT_RATE = 'point_rate'

# DATA STRUCTURE
HEADER = 'header'
HEADER_SHORT = 'h'

TIME = 'time'
TIME_SHORT = 't'

ROTATION = 'rotation'
POSITION = 'position'

POINT = 'point'
FRAME = 'frame'

X = 'x'
Y = 'y'
Z = 'z'
W = 'w'

TYPE = 0
NAME = 1
ID = 2

KEY_SEPARATOR = '-'
SPACE_REPLACEMENT = '_'

# MODALITIES
NONE = 'n'
PLOT_CHART = False

# I/O
READ = 'r'
READ_B = 'rb'
WRITE = 'w'
WRITE_B = 'wb'
APPEND = 'a'
COPY = 'cp'
COMPRESS = 'cmp'
EXPAND = 'exp'
SAVE = 's'
LOAD = 'l'

# EXPERIMENT OPERATIONS
NEW = 0
TEST = 1

# just for developing purposes
NOTHING = -1

# EXPERIMENT PHASES
ALL = 'all'
IMPORT = 'imp'

# LOGGING
# verbose type
OFF = 0
STANDARD = 1
INFO = 2
WARNING = 3
DEBUG = 4

VERBOSE = STANDARD

BASKET = True
