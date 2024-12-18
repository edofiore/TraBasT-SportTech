# Class used only to rise and handle user-defined exceptions
class CustomException(Exception):
    pass

# Function to check if a given string is empty or consists only of
# whitespace characters.
def emptyString(string):
  if string != None and not isinstance(string, str):
    print(f'provided input is not a string')
    return True

  if not (string and string.strip()):
    return True

  return False

# Function that splits the input string into words using split()
# then iterates over each word, extracting the first character of each wordusing
# finally, it joins these first letters together into a single string.
def extractFirstLetters(string):

  # split the string into words
  words = string.split()

  # extract the first letter of each word
  first_letters = [word[0] for word in words]

  # join the first letters into a single string
  result = ''.join(first_letters)

  return result

# function to safely convert user input to an integer
def getIntegerInput(text):
  try:
    # attempt to convert the input to an integer
    value = int(text)
    return value
  except ValueError:
    return -1
