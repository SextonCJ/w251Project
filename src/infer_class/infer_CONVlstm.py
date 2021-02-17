import argparse
import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--record', action='store', default='n', help="enter a value either y or n if you want tor record the video")
parser.add_argument('-m', '--messaging', action='store', default='n', help="enter y if store report to AWS using MQTT")
parser.add_argument('-p', '--path',   action='store', default='../../report', help="directory to record the video, default is home")
parser.add_argument('-f', '--filename',   action='store', default='infer_CONVlstm', help="name of the file, default is infer<datetime>.avi")
parser.add_argument('-c', '--codec',  action='store', default='MJPG', help="recording codec, default is MJPG")
parser.add_argument('-fps', '--fps',    action='store', default='2',   type=int, help="recording frames per second, default is 2")
parser.add_argument('-hg', '--height', action='store', default='640', type=int, help="height of video to record default is 640")
parser.add_argument('-w', '--width',  action='store', default='480', type=int, help="width of video to record default is 640")

args = parser.parse_args()

timestr = time.strftime("%Y%m%d-%H%M%S")
vid_filename = args.path + '/' + args.filename + '_' + timestr + '.avi'
rep_filename = args.path + '/report_' + timestr + '.txt'

# check for recording option
if args.record == 'y':
    print('video saved at: ', vid_filename)
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writerC = cv2.VideoWriter(vid_filename, fourcc, 3, (args.height, args.width))

# make sure we have enough GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

model = '../../models/convLSTM/ConvLSTM2D20201205-112215_65.hdf5'

classifier = load_model(model)
class_labels = {0: 'Really Engaged', 1: 'Doing OK', 2: 'A Bit Bored', 3: 'Checked Out'}

brd0_count = 0
brd1_count = 0
brd2_count = 0
brd3_count = 0

cap = cv2.VideoCapture(0)
cnt = 0
features_arr = []

while True:

    ret, frame = cap.read()
    image = tf.image.convert_image_dtype(image=frame, dtype=tf.float32)
    image = tf.image.resize(images=image,
                            size=[160, 160],
                            method=tf.image.ResizeMethod.BILINEAR,
                            antialias=True).numpy()
    feats = image.reshape(1, 160, 160, 3)

    if cnt<19:
        if cnt == 0:
            features = feats
        else:
            #features = np.append(features, feats)
            features = np.vstack([features, feats])
        cnt+=1
    else:
        features = np.vstack([features, feats])
        #print('*****************************', features)
        pred_features = features[-20:]

        pred_features = pred_features.reshape(1, 20, 160, 160, 3)
        preds = classifier.predict(pred_features)
        label = class_labels[preds.argmax()]
        features = features[1:]
        # label = 'Chris'

        if label == 'Really Engaged':
            brd0_count += 1
        elif label == 'Doing OK':
            brd1_count += 1
        elif label == 'A Bit Bored':
            brd2_count += 1
        elif label == 'Checked Out':
            brd3_count += 1

        # remove this to do update on previous
        #cnt = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2
    x = 50
    y = 50

    cv2.putText(frame, 'Really Engaged: ' + str(brd0_count), (x, y), font, fontScale, (255, 102, 102), thickness)
    cv2.putText(frame, 'Doing OK:' + str(brd1_count), (x, y + 50), font, fontScale, (255, 178, 102), thickness)
    cv2.putText(frame, 'A Bit Bored: ' + str(brd2_count), (x, y + 100), font, fontScale, (178, 102, 255), thickness)
    cv2.putText(frame, 'Checked Out: ' + str(brd3_count), (x, y + 150), font, fontScale, (102, 255, 178), thickness)


    if args.record == 'y':
        writerC.write(frame)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
