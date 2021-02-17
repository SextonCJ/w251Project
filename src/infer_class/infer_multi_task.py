import argparse
import cv2
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import load_model


timestr = time.strftime("%Y%m%d-%H%M%S")
caffe_dir = '../../models/caffe/'

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--record', action='store', default='n', help="enter a value either y or n if you want tor record the video")
parser.add_argument('-p', '--path',   action='store', default='../../report', help="directory to record the video, default is home")
parser.add_argument('-f', '--filename',   action='store', default='infer_multi', help="name of the file, default is infer<datetime>.avi")
parser.add_argument('-c', '--codec',  action='store', default='MJPG', help="recording codec, default is MJPG")
parser.add_argument('-fps', '--fps',    action='store', default='2',   type=int, help="recording frames per second, default is 2")
parser.add_argument('-hg', '--height', action='store', default='640', type=int, help="height of video to record default is 640")
parser.add_argument('-w', '--width',  action='store', default='480', type=int, help="width of video to record default is 640")
parser.add_argument('-m', '--model',  action='store',
                                      default='../../models/multi_task/xception_20201128-191533_10.hdf5',
                                      #default='../../models/multi_task/MobileNetv2_20201205-190003_10.hdf5',
                                      help="model to load")

args = parser.parse_args()

vid_filename = args.path + '/' + args.filename + '_' + timestr + '.avi'
rep_filename = args.path + '/report_multi_' + timestr + '.txt'

if args.record=='y':
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writerC = cv2.VideoWriter(vid_filename, fourcc, 3, (args.height, args.width))

# make sure we have enough GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

classifier = load_model(args.model)
class_labels = ['Boredom', 'Engagement', 'Confusion', 'Frustration']

Boredom_count = 0
Engagement_count = 0
Confusion_count = 0
Frustration_count = 0

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    #image = tf.io.decode_jpeg(contents=frame, channels=3)
    image = tf.image.convert_image_dtype(image=frame, dtype=tf.float32)
    image = tf.image.resize(images=image,
                            size=[299, 299],
                            method=tf.image.ResizeMethod.BILINEAR,
                            antialias=True).numpy()
    image = image.reshape(1, 299, 299, 3)

    preds = classifier.predict(image)

    def get_pred(class_id):
        if np.argmax(preds[class_id]):
            return np.argmax(preds[class_id])
        else:
            return 0

    Boredom = 'Bored: ' + str(get_pred(0))
    Engagement = 'Engaged:' + str(get_pred(1))
    Confusion = 'Confused:' + str(get_pred(2))
    Frustration = 'Frustrated:' + str(get_pred(3))

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2
    x = 50
    y = 50

    cv2.putText(frame, Boredom, (x,y), font, fontScale, (255, 102, 102), thickness)
    cv2.putText(frame, Confusion, (x,y+50), font, fontScale, (255, 178, 102), thickness)
    cv2.putText(frame, Frustration, (x, y + 100), font, fontScale, (178, 102, 255), thickness)
    cv2.putText(frame, Engagement, (x, y + 150), font, fontScale, (102, 255, 178), thickness)

    if args.record == 'y':
        writerC.write(frame)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()

with open(rep_filename, 'w+') as writefile:
    writefile.write('Bored: ' + str(Boredom_count))
    writefile.write('\nEngaged:' + str(Engagement_count))
    writefile.write('\nConfused: ' + str(Confusion_count))
    writefile.write('\nFrustrated: ' + str(Frustration_count))