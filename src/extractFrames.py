import os
import subprocess
import glob
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("fps", help="enter a value either 1 or 2")
args = parser.parse_args()

fps = args.fps
fps_dir = fps + 'FPS/'
fps_val = ' -vf fps=' + fps + '.0 '

def split_video(video_file, image_name, destination_path):
     query = 'ffmpeg -i ' + video_file + fps_val + destination_path + '/' + image_name + '_%d.jpg' 
     #print(query)
     response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()

for train_dir in ['Train/', 'Test/', 'Validation/']:
    source_path = '../data/DAiSEE/DataSet/' + train_dir
    for file_path in Path(source_path).glob('**/*.*'):
        clip = os.path.basename(file_path)[:-4]
        destination_path = '../data/DAiSEE/' + fps_dir + 'dataImages/' + train_dir
        destination = os.path.dirname(destination_path)
        file_path = str(file_path)
        split_video(file_path, clip, destination)
