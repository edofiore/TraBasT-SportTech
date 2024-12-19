from global_constants import *
from I_O import readData
from models import createModels
from plotter import plotData
from data_manipulation import filterData
# from main import choice

# This class is a structured framework that guides the user through the various stages of conducting an experiment,
# from data input to model processing and visualization. It enforces consistency through controlled property access
# and method-based operations, ensuring that each step of the experiment is executed correctly and in the proper sequence.
class Experiment():

  def __init__(self, operation=NEW, inFile='', outPath='',
              saveData=False, savePath='', verbose=INFO):

    self._status = False
    self._operation = operation

    self._inFile = inFile
    self._outPath = outPath

    self._saveData = saveData
    self._savePath = savePath

    self._verbose = verbose

    self._description = {}

    self._models = []

    self.initialize()

  @property
  def status(self):
    return self._status

  @status.setter
  def status(self, s):
    print(f'Error: Invalid operation, experiment status cannot be manually modified\n')

  @property
  def operation(self):
    return self._operation

  @operation.setter
  def operation(self, o):
    print(f'Error: Invalid operation, experiment operation cannot be manually modified\n')

  @property
  def inFile(self):
    return self._inFile

  @inFile.setter
  def inFile(self, file):
    print(f'Error: Invalid operation, experiment input file cannot be manually modified\n')

  @property
  def outPath(self):
    return self._outPath

  @outPath.setter
  def outPath(self, path):
    print(f'Error: Invalid operation, experiment output directory cannot be manually modified\n')

  @property
  def verbose(self):
    return self._outPath

  @verbose.setter
  def verbose(self, val):
    print(f'Error: Invalid operation, experiment verbose mode cannot be manually modified\n')

  @property
  def description(self):
    return self._description

  @description.setter
  def description(self):
    print(f'Error: Invalid operation, experiment description cannot be manually modified\n')

  @property
  def models(self):
    return self._models

  @models.setter
  def models(self):
    print(f'Error: Invalid operation, experiment models cannot be manually modified\n')

  @property
  def saveData(self):
    return self._saveData

  @saveData.setter
  def saveData(self, save):
    if save != self._saveData:
      self._saveData = save

    return self._saveData

  @property
  def savePath(self):
    return self._savePath

  @savePath.setter
  def savePath(self, path):
    if emptyString(path):
      print(f'Error: Given save directory is invalid: {path}\n')
      return False
    else:
      self._savePath = path

    return True

  def abort(self):
    self.__erase()
    print(f'Experiment aborted')

  # It initializes the experiment and performs necessary operations
  # based on the specified parameters.
  def initialize(self):

    print(f'Initializing the experiment\n')
    if self.checkData():
        self._status = True

    if self._status:
      print(f'Experiment correctly initialized\n')
    else:
      self.abort()

  # It initializes the description of the experiment.
  def initializeDescription(self, data):

    self._description = dict(data[HEADER][HEADER_SHORT])

  # It verifies various aspects of the input data:
  #   - input file
  #   - output path
  #   - save path
  # to ensure that they are valid and exist.
  def checkData(self):
    if emptyString(self._inFile):
      print(f'Error: Invalid or empty input file: {self._inFile}\n')
      return False

    if not os.path.isfile(self._inFile):
      print(f'Error: Invalid input file: {self._inFile}\n')
      return False

    fName, fExt = os.path.splitext(self._inFile)
    fExt = fExt.lower()
    if fExt not in INPUT_EXTENSIONS:
      print(f'Error: Invalid input file extension: {fExt}\n')
      return False

    if not emptyString(self._outPath):
      if not os.path.isdir(self._outPath):
        print(f'Error: Invalid output path: {self._outPath}\n')
        return False
    else:
      # it can be made dependent on the type of the operation
      if self._verbose >= INFO:
        print(f'Info: Output path empty\n')

    if self._saveData:
      if not emptyString(self._savePath):
        if not os.path.isdir(self._savePath):
          print(f'Error: Invalid save path: {self._savePath}\n')
          return False
      else:
        print(f'Error: Invalid or empty save path:{self._savePath} \n')
        return False

    return True

  # It resets all the attributes of the Experiment class,
  # clearing any previous state and preparing it for a new experiment.
  def __erase(self):

    self._status = False
    self._operation = NEW

    self._inFile = ''
    self._outPath = ''

    self._saveData = False
    self._savePath = ''

    self._verbose = INFO

    self._description = {}

    self._models = []

  # It is responsible for importing data from a specified input file.
  def importData(self):

    fName = os.path.basename(self._inFile)
    print(f'Importing data data from {fName} file\n')

    data = readData(self._inFile)

    if data is not None:
      print(f'Data correctly imported\n')
    else:
      print(f'Error: Something wrong reading the data\n')

    return data

  # It is responsible for formatting data and creating models.
  # It involves some preprocessing of the data.
  def prepareModels(self, data):
    
    print(f'Preparing data\n')

    self._models = createModels(self._inFile, data)

    print(f'Model(s) correctly prepared\n')

  def plotModelsData(self, models=None):

    print(f'Plotting data\n')
    COMPARE = False
    
    selfCopy = copy.deepcopy(self)
    modelsCopy = copy.deepcopy(models)
    if models is None:
      option = '-2'
      while True:
          print(f'Do you want to compare this performance with the one of someone else?')
          print(f'1. Yes')
          print(f'0. No, continue with visualization of the single performance.')

          option = input(f'Enter your choice: ')
          if option == '1': # it was chosen to compare the performance
            COMPARE = True
            option, srcPath = choice(
            message='Please select the performance to compare:',
            lastLevel=False,
            params=CSV_LIST)
            if option not in ['0', 'r']:
              srcPath = os.path.join(DATA_PATH, srcPath)
              print(f'File: {srcPath}')
              if srcPath == self._inFile:
                print(f'You are comparing the same performance, please select another one\n')
                continue
              # we chose the performance to compare, now we process the data
              if self.checkData():
                self._inFile = srcPath
              else:
                print(f'Error: Invalid input file: {srcPath}\n')
                break
              dataToCompare = self.importData()
              if dataToCompare is not None:
                self.prepareModels(dataToCompare)
                if len(self._models) > 0:
                  if BASKET:
                      print(f'Do you want to apply the Kalman filter to the data?')
                      print(f'1. Yes')
                      print(f'0. No')
                      option = input(f'Enter your choice: ')
                      if option == '1': # we apply the Kalman filter
                        self.filterModelsData()
                        break
                      elif option == '0': # no kalman filter
                        print(f'No filtering applied\n')
                        break
                  else: # no basket
                    self.filterModelsData()
                    break
              else: # dataToCompare is None
                print(f'Error: Something wrong reading the data\n')
                return
          elif option == '0': # no comparison
            break
      if COMPARE:
        plotData(selfCopy._inFile, selfCopy._models, self._inFile, self._models, COMPARE)
      else:
        plotData(self._inFile, self._models)
          
      
    else:
      option = '-2'
      while True:
          print(f'Do you want to compare this performance with the one of someone else?')
          print(f'1. Yes')
          print(f'0. No, continue with visualization of the single performance.')

          option = input(f'Enter your choice: ')
          if option == '1':
            COMPARE = True
            option, srcPath = choice(
            message='Please select the performance to compare:',
            lastLevel=False,
            params=CSV_LIST)
            if option not in ['0', 'r']:
              srcPath = os.path.join(DATA_PATH, srcPath)
              print(f'File: {srcPath}')
              if srcPath == self._inFile:
                print(f'You are comparing the same performance, please select another one\n')
                continue
              # we chose the performance to compare, now we process the data
              if self.checkData():
                self._inFile = srcPath
              else:
                print(f'Error: Invalid input file: {srcPath}\n')
                break
              dataToCompare = self.importData()
              if dataToCompare is not None:
                self.prepareModels(dataToCompare)
                if len(self._models) > 0:
                  if BASKET:
                    while True:
                      print(f'Do you want to apply the Kalman filter to the data?')
                      print(f'1. Yes')
                      print(f'0. No')
                      option = input(f'Enter your choice: ')
                      if option == '1':
                        self.filterModelsData()
                        break
                      elif option == '0':
                        print(f'No filtering applied\n')
                        break
                  else: # no basket
                    self.filterModelsData()
              else: # dataToCompare is None
                print(f'Error: Something wrong reading the data\n')
                return
          elif option == '0': # no comparison
            break
      if COMPARE:
        plotData(selfCopy._inFile, modelsCopy, self._inFile, models, COMPARE)
      else:
        plotData(self._inFile, models)

  def filterModelsData(self):

    print(f'Filtering data\n')

    filterData(self._inFile, self._models)

  # It ensures the sequential execution of the experiment steps and provides
  # a convenient way to run the entire experiment with a single method call.
  def run(self):
    if self._operation == NEW:
      data = self.importData()

      if data is not None:
        self.prepareModels(data)

        if len(self._models) > 0:
          if BASKET:
            while True:
              print(f'Do you want to apply the Kalman filter to the data?')
              print(f'1. Yes')
              print(f'0. No')
              option = input(f'Enter your choice: ')
              if option == '1':
                self.filterModelsData()
                break
              elif option == '0':
                print(f'No filtering applied\n')
                break
          else:
            self.filterModelsData()
          
          self.plotModelsData()

          return True

      return True

    if data is None:
      self.abort()

    return False

  # It provides a convenient way to quickly view the relevant information about
  # the experiment configuration and current state.
  def info(self):
    print(f"Experiment info:")
    print(f"\tStatus: {self._status}")
    print(f"\tOperation: {self._operation}")
    print(f"\tInput file: {self._inFile}")
    print(f"\tOutput directory path: {self._outPath}")
    print(f"\tSave experiment: {'True' if self._saveData else 'False'}")
    print(f"\tSave directory path: {self._savePath}")
    print(f"\tVerbose level: {self._verbose}\n")

    for key, value in self._description.items():
      print(f"\t{key.replace(SPACE_REPLACEMENT, ' ')}: {value}")

    for model in self._models:
      print()
      model.info()

def choice(message, lastLevel, params):

  str = ''
  option = '-2'

  while option not in ['0']:
    print(message)

    for i in range(len(params)):
      print(f'{i+1}. {params[i]}')

    print(f'{len(params)+1}. Return to the previous menu')
    print(f'0. Exit the program')

    option = input('Enter your choice: ')
    val = getIntegerInput(option)

    if val == 0:
      print(f'Exiting the program...')
      break
    elif val == len(params)+1:
      option = 'r'
      break
    elif val > 0 and val <= len(params):
      tmpStr= ''
      str = params[val-1]

      if option != 'r':
        str = str if emptyString(tmpStr) else  os.path.join(tmpStr, str)		
        break

    else:
      print(f'Invalid input, try again.')

  return option, str