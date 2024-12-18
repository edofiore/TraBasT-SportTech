from global_constants import *

# Function to create instances of different model types based on the 
# file extension and data provided.
# It acts as a factory method encapsulationg the logic for creating models of different types
# and allowing for easy extension to support additional file formats or model types in the future.
def createModels(filePath=None, data=None):
    
    if data is None or filePath is None:
        print(f'Error: Invalid data or file path\n')
        return None

    models = []

    _, fExt = os.path.splitext(filePath)
    
    if fExt == EXTENSIONS[Extension.csv]:

        if BASKET:
            for i in range (0, len(data), 3):
                model = CSVModel(data[i:i+3], None)
                models.append(model)
        else:
            for item in data.items():
                key, value = item  # Unpack the tuple into key and value
                if key != HEADER:
                    model = CSVModel(item, header=data[HEADER])
                    models.append(model)

    elif fExt == EXTENSIONS[Extension.bvh]:
        model = BVHModel(data)
        models.append(model)

    elif fExt == EXTENSIONS[Extension.c3d]:
        model = C3DModel(data)
        models.append(model)

    else:
        print(f'Error: Invalid file extension: {fExt}\n')
        return []

    return models if len(models) > 0 else []

# Class designed to represent and provide information about a model 
# derived from CSV data by encapsulating functionality for computing 
# different operations on the data, such as initializing the model
# or getting informative details about it.
class CSVModel:

    def __init__(self, data=None, header=None):

        self._description = {}

        self._time = [] # seconds
        self._rotations = {}
        self._positions = {}
        self._kfPositions = {}
        self._splPositions = {}

        if data is not None:
            self.initialize(data, header)

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, s):
        print(f'Error: Invalid operation, model description cannot be manually modified\n')

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        print(f'Error: Invalid operation, model time dictionary cannot be manually modified\n')

    @property
    def rotations(self):
        return self._rotations

    @rotations.setter
    def rotations(self, s):
        print(f'Error: Invalid operation, model rotations dictionary cannot be manually modified\n')

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, s):
        print(f'Error: Invalid operation, model positions dictionary cannot be manually modified\n')

    @property
    def kfPositions(self):
        return self._kfPositions

    @kfPositions.setter
    def kfPositions(self, s):
        self._kfPositions = s

    @property
    def splPositions(self):
        return self._splPositions

    @splPositions.setter
    def splPositions(self, s):
        self._splPositions = s

    def getName(self):
        return self._description['Name']

    def getType(self):
        return self._description['Type']

    def getID(self):
        return self._description['ID']

    def isMarker(self):
        return self._description['Type'] == "Marker"

    # It initializes the data object.
    def initialize(self, data, header=None):

        if not BASKET:
            if header is not None:
                self.initializeDescription(header)

            outerKey, dataDict = data  # Unpack the tuple into key and value
            parts = outerKey.split(KEY_SEPARATOR)
            for i, part in enumerate(parts):

                if i == TYPE:
                    self._description['Type'] = part.replace(SPACE_REPLACEMENT, ' ')
                    if "marker" in self._description['Type'].lower():
                        self._description['Type'] = "Marker"

                elif i == NAME:
                    self._description['Name'] = part.replace(SPACE_REPLACEMENT, ' ')
                    if self._description['Name'].find(":") != -1:
                        self._description['Name'] = self._description['Name'].split(":")[1]

                elif i == ID:
                    self._description['ID'] = part.replace(SPACE_REPLACEMENT, ' ')

            for middleKey, value in dataDict.items():

                if middleKey == TIME:
                    self._time = copy.deepcopy(value[TIME_SHORT])
                elif middleKey == ROTATION:
                    self._rotations = copy.deepcopy(value)
                elif middleKey == POSITION:
                    self._positions = copy.deepcopy(value)
        else:
            self._description = {'Name' : data[0][0][:-2]}
            self._description['Type'] = 'Bone'
            self._positions = {X: list(data[0][1:]), Y: list(data[1][1:]), Z: list(data[2][1:])}
            # self._positions = {}
        #     data
        #     self._description = 

    # It initializes the description of the experiment.
    def initializeDescription(self, data):
        
        self._description = dict(data[HEADER_SHORT])

    # It provides a convenient way to quickly view the relevant information
    # about the configuration of the model.
    def info(self):

        print(f"Model info:")

        for key, value in self._description.items():
            print(f"\t{key.replace(SPACE_REPLACEMENT, ' ')}: {value}")

        print(f"\n\tTime number of elements: {len(self._time)}")

        print(f"\n\tRotations number of elements: {len(self._rotations)}")
        for key, value in self._rotations.items():
            print(f"\t{key.replace(SPACE_REPLACEMENT, ' ')}: {len(value)}")

        print(f"\n\tPositions number of elements: {len(self._positions)}")
        for key, value in self._positions.items():
            print(f"\t{key.replace(SPACE_REPLACEMENT, ' ')}: {len(value)}")

# Class designed to represent and provide information about a model 
# derived from BVH data by encapsulating functionality for computing 
# different operations on the data, such as initializing the model
# or getting informative details about it.
class BVHModel:

    def __init__(self, data):

        self._animation = None
        self._names = None
        self._frameTime = None

        self.initialize(data)

    # It initializes the data object.
    def initialize(self, data):

        for key, value in data.items():

            if key == ANIMATION:
                self._animation = value
            elif key == NAMES:
                self._names = value
            elif key == TIME:
                self._frameTime = value

    # It provides a convenient way to quickly view the relevant information
    # about the configuration of the model.
    def info(self):

            print(f"Model info:")

            print(f"\n\tNames number of elements: {len(self._names)}")
            print(f"\n\tFrame time: {self._frameTime}")

# Class designed to represent and provide information about a model 
# derived from C3D data by encapsulating functionality for computing 
# different operations on the data, such as initializing the model
# or getting informative details about it.
class C3DModel:

    def __init__(self, data):

        self._pointRate = 0
        self._scaleFactor = 0
        self._frames = {}

        self.initialize(data)

    # It initializes the data object.
    def initialize(self, data):

        self._pointRate =data[C3D_POINT_RATE]
        self._scaleFactor = data[C3D_SCALE_FACTOR]
        self._frames = data[FRAME]

    # It provides a convenient way to quickly view the relevant information
    # about the configuration of the model.
    def info(self):

        print(f"Model info:")

        print(f"\n\tPoint rate: {self._pointRate}")
        print(f"\n\tScale factor: {self._scaleFactor}")
        print(f"\n\tFrames number of elements: {len(self._frames)}")
