# TraBasT-SportTech

TraBasT (Tracked Basketball Training) is a university project of the Sport-Tech course in University of Trento. 
The concept behind this project is to use motion capture to track basketball players movement during a free throw. The final goal is to help amateur players learn and improve their free throws by comparing their free throw motions against a gold standard and giving them performance based feedback.

# Data
We put already some data inside code->Data. If you want to add your own data to analyze, add them to the Data folder. Make sure that they are in CSV format and that they are organized in the same fashion as the one already inside the folder. 

# Starting the project
Starting this project is very easy:
1) install the necessary libraries in requirements.txt
2) go to the main.py file and run it

After running main.py, an interactive menu will appear. It is really easy to follow, it will guide you in the next steps. You can plot a single player shooting or plot two players shooting. The secod player/file selected will be the gold standard, the first will be the player you want to evaluate. After selecting the two player option, you will receive the evaluation of the player. After that, you will be asked if you want to save the video that is about to be showed. You can change the fps per second of the saved video by changing the FPS variable at the top of the plotter.py file. Then finally the video will show.

