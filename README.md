# Engagement Detection from Video Capture

## Homework Submission = __Whitepaper.pdf__

## Folder and File Structure
The most important files / directories are in bold. data and models are not included due to size. Data used for training can be requested from [DAiSEE](https://iith.ac.in/~daisee-dataset/)

GazeML code referenced in the appendix is available from [github](https://github.com/swook/GazeML/blob/master/src/elg_demo.py).

### This Directory

- __Whitepaper.pdf__  
PDF describing detailed methodology for project

- __aws_setup.md & jetson_setup.md__  
step by step instructions on setting up aws and jetson for this project, including docker implementation, library installs and OpenCV compilation.

- Presentation.pdf     
In class presentaion

- EDA.xls  
Spreadsheet containing modeling results and some EDA

- tree.txt
directory structure

### src
Contains all code for the Project

- __cnn__  
Jupyter notebooks (in order) for the cnn code used to get data, organize and train models.

- __rnn__  
Jupyter notebooks (in order) for the two LSTM models

- __infer_class__  
Scripts to run inference. __infer_dnn.py__ has the most complete code (argparse options, and MQTT)

- extract_frames.py  
code to extract frames from videos, pass in FPS as integer argument

### report
Output files from running demos / inference

- __infer_output__  
Contains subdirectories for each inference script, containing videos recorded inference/demo programs and report from dnn model (CNN)

- resuts_images   
Contains images of classification matrices referenced to experiments described in EDA.xlsx

- gazeML.png  
image extracted from runnign gaze_ml demo

### messaging
Scripts to setup MQTT messaging on AWA and Jetson, using docker.
