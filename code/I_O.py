import csv
import c3d # if library not found: pip install c3d

from global_constants import *

folderPath = os.path.join(ROOT_PATH, BHV_DIR)
sys.path.append(folderPath)

# from BVH_FILE import *

# Function to handle directory-related checks and validation based on
# the specified mode and path parameters.
# It returns True if the checks pass and False if any errors or
# invalid conditions are encountered.
def checkDir(mode=None, path=None):

  if mode == None:
    print(f"Error: Invalid selected operation\n")
    return False

  if emptyString(path):
    print(f"Error: Invalid path: {path}\n")
    return False

  if mode == SAVE:
    if not os.path.isdir(path):
      os.makedirs(path)

    True if os.path.isdir(path) else False

  elif mode == LOAD:
    if not os.path.isdir(path):
      print(f"Error: Specified path doesn't exist, impossible to load data\n")
      return False

    if os.listdir(path) == []:
      print(f"Error: {path} doens't contain any file\n")
      return False

  return True

# Function to handle various checks and validations related to files, based
# on the specified mode and file path parameters.
# It returns True if the checks pass, returns the new file path in
# case of saving mode, and returns False if any errors or invalid
# conditions are encountered.
def checkFile(mode=None, filePath=None):

  if mode == None:
    print(f"Error: Invalid selected operation\n")
    return False

  if emptyString(filePath):
    print(f"Error: Invalid file: {filePath}\n")
    return False

  if os.path.exists(filePath):
    if os.path.isfile(filePath):

      if mode == SAVE:
        found = False

        for i in range(2, 11):
          newFilePath, extention = os.path.splitext(filePath)

          if i > 2:
            newFilePath, _ = newFilePath.rsplit('_', 1)

          newFilePath += (f'_{i}')
          filePath = newFilePath + extention

          if VERBOSE >= WARNING:
            print(f"extention: {extention}")
            print(f"newPathFile: {newFilePath}")
            print(f"filePath: {filePath}")

          if not os.path.exists(filePath):
            found = True
            break

        if not found:
          print(f"Error: Impossible to create file: {filePath}, and save data\n")
          return False

        return filePath

      elif mode == LOAD:
        return True

    else:
      print(f"Error: Passed string isn't a file: {filePath}")
      print(f"Impossible to save/load data\n")
      return False

  elif mode == SAVE:
    return True
  else:
    print(f"Error: File doesn't exist: {filePath}, impossible to load data\n")
    return False

  return True

# Function for reading data from different file types
# based on the specified file type.
# It returns the imported data if successful
# or None if unsuccessful.
def readData(filePath=None):

  res = checkFile(mode=LOAD, filePath=filePath)
  if isinstance(res, bool) and not res:
    return None

  _, fExt = os.path.splitext(filePath)

  data = None
  if fExt == EXTENSIONS[Extension.csv]:
    print(f'CSV file reading')
    data = readCSV(filePath)

  elif fExt == EXTENSIONS[Extension.bvh]:
    print(f'BVH file reading')
    data = readBVH(filePath)

  elif fExt == EXTENSIONS[Extension.c3d]:
    print(f'C3D file reading')
    data = readC3D(filePath)

  else:
    print(f'Error: Invalid file extension: {fExt}\n')

  return data

# Function to read the CSV file at the specified location and returns
# the data as a list of lines.
# If the file exists, can be read, and contains any lines,
# the data list is returned.
# If the CSV file is empty or cannot be read, None is returned.
def readCSV(filePath):

  data = []
  try:

    with open(filePath, READ) as f:
      for line in csv.reader(f):
        data.append(line)

      if data is not None:
        data = extractDataCSV(data)

  except Exception as e:
    print(f'Error: Impossible to read CSV file - {e}\n')
    return None

  if VERBOSE >= DEBUG:
    for line in data:
      print(f'{line}')

  return data if len(data) > 0 else None

# Function to reads data from a BVH (Biovision Hierarchy) file located
# at the given file path.
# It is a function wrapper around the read_bvh() function provided by
# an external library.
# The returned data are organized into a dictionary.
# This last is returned if the reading process is successful.
# If an error occurs during the reading process, None is returned.
def readBVH(filePath):

  data = {}
  try:

    animation, names, frameTime = read_bvh(file_name=filePath)

    data[ANIMATION] = animation
    data[NAMES] = names
    data[TIME] = frameTime

  except Exception as e:
    print(f'Error: Impossible to read BVH file - {e}\n')
    return None

  if VERBOSE >= DEBUG:
    for key, value in data.items():
        print("Key:", key)
        print("Type of Value:", type(value))

  return data

# Function to read data from a C3D (Coordinate 3D) file
# located at the given filePath.
# It returns the data dictionary if at least one frame
# was successfully read, otherwise it returns None.
#  If an error occurs during the process returns None.
def readC3D(filePath):

  data = None
  try:

    with open(filePath, READ_B) as f:
        dataReader = c3d.Reader(f)

        data = extractDataC3D(dataReader)

  except Exception as e:
    print(f'Error: {e}')
    return None

  return data

# Function to extract and process data from a pre-acquired CSV file.
# It handles header extraction, version checking, time extraction, data processing,
# and error handling.
# Finally, it returns a dictionary containing the extracted data or None in case of errors.
def extractDataCSV(data):

    times = []
    # extract the header of the file and check if the version is compatible
    
    # If we have the BASKET csv version
    if BASKET:
      try:
        # the file is structured in a way that the first row of each column has the name 
        # of the bone and its coordinate, for example Hips_x, Hips_y, Hips_z and the other cells 
        # of the column contain the value of the coordinate in one frame
        for row in data[:CSV_HEADER_FILE_LEN]:
          for i in range(0, len(row), 2):
            if row[i] != '':
              key = row[i].replace(' ', SPACE_REPLACEMENT)
              # if there aren't 3 columns for each bone, the file is not supported
              if (i!=0 and (i+1)%3 != 1) and (row[i][:-2] != row[i-1][:-2]):
                raise CustomException('Error: CSV file version is not supported\n')

        # save the information about the time
        # which will be common for all
        # in the csv file, the time is always the column before the data
        # convert the string to float
        for j, row in enumerate(data[1:]):
          for i, coordinate in enumerate(row):
            if coordinate == '':
              data[j+1][i] = 0
            else:
              data[j+1][i] = float(coordinate)

        # transpose the data 
        data = list(zip(*data))
        dataDict = data

                      
      except Exception as e:
          print(f'Error: Impossible to extract data from CSV file - {e}\n')
          dataDict = None
    
    else:
      # If we have the SQUAT OR CV csv version
      dataDict = {HEADER: {HEADER_SHORT: {}}}
      try:
          # the file is structured in a way that the even columns contain the 
          # description of the data, while the odd columns contain the data
          for row in data[:CSV_HEADER_FILE_LEN]:
              for i in range(0, len(row), 2):
                  if row[i] != '':
                      key = row[i].replace(' ', SPACE_REPLACEMENT)
                      value = row[i + 1]

                      if (i+1) == CSV_VERSION_COLUMN and value != CSV_VERSION:
                          raise CustomException(f'Error: CSV file version {value} is not supported\n')

                      dataDict[HEADER][HEADER_SHORT][key] = value


          # remove the header of the file
          # we don't need it anymore
          for i in range(CSV_HEADER_DATA_ROW):
              del data[0]

          # save the information about the time
          # which will be common for all
          # in the csv file, the time is always the comlumn before the data
          # convert the string to float
          for row in data[CSV_HEADER_DATA_LEN:]:
              times.append(0 if row[CSV_DATA_COLUMN-1] == '' else float(row[CSV_DATA_COLUMN-1]))

          # transpose the data and remove the first two rows
          # we don't need them anymore
          data = list(zip(*data))
          for i in range(CSV_DATA_COLUMN):
              del data[0]

          # create a dictionary of dictionaries with the data
          # the first key is the combination of the first CSV_HEADER_DATA_LEN-1 elements
          # of the header; -1 because we want to use the last element of the header
          # as the key for the sub-dictionary
          for row in data:

              # CSV file can contain some data that we want to ignore
              # we can specify the type of data to ignore in the IGNORE_DATA list
              if row[TYPE] in IGNORE_DATA:
                  continue

              outerKey = ''
              middleKey = ''
              innerKey = ''
              for i in range(CSV_HEADER_DATA_LEN):

                  if i < CSV_HEADER_DATA_LEN-2:
                      tmpStr = row[i].replace(' ', SPACE_REPLACEMENT)
                      outerKey += tmpStr
                      outerKey = outerKey + KEY_SEPARATOR if i < CSV_HEADER_DATA_LEN-3 else outerKey

                  elif i < CSV_HEADER_DATA_LEN-1:
                      tmpStr = row[i].replace(' ', SPACE_REPLACEMENT)
                      middleKey += tmpStr.lower()

                  else:
                      innerKey = row[CSV_HEADER_DATA_LEN-1] if row[CSV_HEADER_DATA_LEN-1] != '' else extractFirstLetters(row[CSV_HEADER_DATA_LEN-2])
                      innerKey = innerKey.lower()
                      values = [float(x) if x != '' else float(0) for x in row[CSV_HEADER_DATA_LEN:]]

              if outerKey in dataDict:
                  if middleKey in dataDict[outerKey]:
                      dataDict[outerKey][middleKey][innerKey] = values
                  else:
                      dataDict[outerKey][middleKey] = {innerKey: values}
              else:
                  dataDict[outerKey] = {middleKey: {innerKey: values}}

          # finally add to each middle dictionary the time
          for outerKey, middleDict in dataDict.items():
              if outerKey != HEADER:
                  middleDict[TIME] = {TIME_SHORT: times}

          if VERBOSE >= DEBUG:
            # printing the structure
            for outerKey in sorted(dataDict.keys()):
                print(outerKey + ':')
                middleDict = dataDict[outerKey]
                for middleKey in sorted(middleDict.keys()):
                    print('\t' + middleKey + ':')
                    innerDict = middleDict[middleKey]
                    for innerKey in sorted(innerDict.keys()):
                        print('\t\t' + innerKey + ':')#, innerDict[innerKey][0:3])

      except Exception as e:
          print(f'Error: Impossible to extract data from CSV file - {e}\n')
          dataDict = None

    return dataDict

# Function to extract data from a C3D file using a provided dataReader
# capable of reading C3D files.
# It extracts point rate, scale factor, and frames data.
# If the extraction process is successful, it returns a dictionary containing the data.
# Otherwise, it returns None.
def extractDataC3D(dataReader):

    data = {C3D_POINT_RATE: 0, C3D_SCALE_FACTOR : 0, FRAME : {}}
    try:

        data[C3D_POINT_RATE] = dataReader.point_rate
        point_scale = abs(dataReader.point_scale)
        data[C3D_SCALE_FACTOR] = point_scale
        for i, points, analog in dataReader.read_frames():

            frameData = {}
            for (x, y, z, err_est, cam_nr), label in zip(points,
                                        dataReader.point_labels):

                label = label.strip()
                frameData[label] = {
                    X : x * point_scale,
                    Y : y * point_scale,
                    Z : z * point_scale,
                    C3D_ERR_EST: err_est,
                    C3D_CAMERA_NR: cam_nr
                }

            data[FRAME][i] = frameData

    except Exception as e:
        print(f'Error: Impossible to extract data from C3D file - {e}')
        return None

    return data
