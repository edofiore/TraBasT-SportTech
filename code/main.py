from experiment import *

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

# This function puts all the modules together to import and process the data
def main(operation=NEW, phase=ALL, outPath=''):

  option = '-2'
  srcPath = ''

  if operation == TEST:
    option = 't'
    srcPath = SRC_PATH

  while option not in ['0', 't']:
      print('Please select the operation you want to perform:')
      print('1. Import and process data - CSV files')
      print('0. Exit the program')

      option = input('Enter your choice: ')

      if option == '1':
        option, srcPath = choice(
          message='Please select the file on which you want to perform the operations:',
          lastLevel=False,
          params=CSV_LIST)

        if option not in ['0', 'r']:
          break

      elif option == '0':
          print('Exiting the program...')
      else:
          print('Invalid input, try again.')

      print()

  if option == '0':
      return True
  else:
    srcPath = os.path.join(DATA_PATH, srcPath)
    print(f'File: {srcPath}')

  # new experiment
  if operation == NEW or operation == TEST:
    experiment = Experiment(operation=operation,
                            inFile=srcPath,
                            outPath=outPath,
                            verbose=DEBUG
                            )

  else:
    print(f'Error: Invalid operation: {operation}')
    print(f'Allowed operations are:')
    print(f'\t - {NEW} for running a new experiment')
    return False

  if not experiment.status:
    print(f'Error: Something wrong with the experiment initialization')
    return False

  if phase == ALL:
    if experiment.run():
      pass
  else:
    print(f'Error: Selected an invalid experiment phase: {phase}')
    print(f'Allowed phases are:')
    print(f'\t - {ALL} for running all phases of the experiment')

    return False

  return True

main()
