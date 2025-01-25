# TraBasT-SportTech

**TraBasT (Tracked Basketball Training)** is a university project of the Sport-Tech course in University of Trento. 

The concept behind this project is to use motion capture to track basketball players movement during a free throw. The final goal is to help amateur players learn and improve their free throws by comparing their free throw motions against a gold standard and giving them performance-based feedback.

## Data
We have already included some data in the `code/Data` folder. If you want to add your own data for analysis, place it in the `Data` folder. Make sure it is in CSV format and organized in the same way as the files already in the folder.

## Starting the Project
Getting started with this project is very straightforward:

1. **(Optional)** Use a virtual environment to manage dependencies. To create and activate a virtual environment on **Windows**, run:

    * Create the virtual environment (you can personalize the environment name, e.g., my_env):
        ```bash
        python -m venv name_env  # You can personalize the environment name
        ```
    * Activate the virtual environment:
        ```bash
        .\name_env\Scripts\activate
        ```
    For **Linux** or **macOS**, refer to the [official guide for installing and using virtual environments](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

    Alternatively, you can install the dependencies directly on your system.

2. Install the required libraries specified in `requirements.txt` by running the following command:
   ```bash
   pip install -r requirements.txt
   ````

3. Navigate to the `code/main.py` file and run it:

   ```bash
   python main.py
   ```

After running `main.py`, an interactive menu will appear. The menu is very user-friendly and will guide you through the next steps. You can choose to:

- Track the movement of a single player.
- Compare the free throws of two players. In this case, the second player/file selected will be treated as the gold standard, while the first will be the player you want to evaluate.

After selecting the two-player option, you will receive an evaluation of the player. 

Then, you will be asked if you want to save the video that will be displayed. You can change the frames per second (FPS) of the saved video by modifying the `FPS` variable at the top of the `plotter.py` file. 

Finally, the video will be displayed.

## Additional Notes
- Ensure that Python is installed on your system (recommended version: >=3.8).
- If you encounter issues with the packages or running the code, verify that all dependencies listed in `requirements.txt` have been correctly installed.

## Report
The report (a brief explanation of the project and how it is done) can be read [here](https://github.com/edofiore/TraBasT-SportTech/blob/main/Sport_Tech_Report.pdf)
