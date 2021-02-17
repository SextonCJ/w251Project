import os
import subprocess
import glob
from pathlib import Path

path = 'videos/'

def split_video(video_file, image_name, destination_path):
     query = 'ffmpeg -i ' + video_file + ' -vf fps=2.0  ' + destination_path + '/' + image_name + '_%d.jpg' 
     # print(query)
     response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()

for file_path in Path(path).glob('**/*.*'):
    clip = os.path.basename(file_path)[:-4]
    destination = os.path.dirname(file_path)
    file_path = str(file_path)
    split_video(file_path, clip, destination)
