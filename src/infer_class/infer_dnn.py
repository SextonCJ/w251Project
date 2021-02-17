import argparse
import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# create function for callback
def on_publish(client, userdata, result):
    print("data published \n")
    pass

timestr = time.strftime("%Y%m%d-%H%M%S")
caffe_dir = '../../models/caffe/'
model = '../../models/dataFacesAug/MobileNetV2_20201205-191236_6.hdf5'

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--record', action='store', default='n', help="enter a value either y or n if you want tor record the video")
parser.add_argument('-m', '--messaging', action='store', default='n', help="enter y if store report to AWS using MQTT")
parser.add_argument('-p', '--path',   action='store', default='../../report', help="directory to record the video, default is home")
parser.add_argument('-f', '--filename',   action='store', default='infer', help="name of the file, default is infer<datetime>.avi")
parser.add_argument('-c', '--codec',  action='store', default='MJPG', help="recording codec, default is MJPG")
parser.add_argument('-fps', '--fps',    action='store', default='2',   type=int, help="recording frames per second, default is 2")
parser.add_argument('-hg', '--height', action='store', default='640', type=int, help="height of video to record default is 640")
parser.add_argument('-w', '--width',  action='store', default='480', type=int, help="width of video to record default is 640")

args = parser.parse_args()
vid_filename = args.path + '/' + args.filename + '_' + timestr + '.avi'
rep_filename = args.path + '/report_' + timestr + '.txt'

# check for recording option
if args.record == 'y':
    print('video saved at: ', vid_filename)
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writerC = cv2.VideoWriter(vid_filename, fourcc, 3, (args.height, args.width))

# check for messaging option, create mqtt client
if args.messaging == 'y':
    import paho.mqtt.client as mqtt
    LOCAL_MQTT_HOST = "mosquitto"
    LOCAL_MQTT_PORT = 1883
    LOCAL_MQTT_TOPIC = "jetson/report"

    mqtt_client = mqtt.Client()
    mqtt_client.on_publish = on_publish
    mqtt_client.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT)

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

# load model from disk
net = cv2.dnn.readNetFromCaffe(caffe_dir + 'deploy.prototxt.txt', caffe_dir + 'res10_300x300_ssd_iter_140000.caffemodel')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classifier = load_model(model)

class_labels = {0: 'Really Engaged', 1: 'Doing OK', 2: 'A Bit Bored', 3: 'Checked Out'}

brd0_count = 0
brd1_count = 0
brd2_count = 0
brd3_count = 0

def face_detector(image):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300,300), (103.93, 116.77, 123.68))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            roi_face = cv2.resize(face, (224, 224), interpolation = cv2.INTER_AREA)
            return (startX, startY, endX, endY), roi_face, image
        else:
            (startX, startY, endX, endY) = (0,0,0,0)
            roi_face = np.zeros((224,224), np.uint8)
            return (startX, startY, endX, endY), roi_face, image
         
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    rect, face, image = face_detector(frame)

    if np.sum([face]) != 0.0:
        roi = face.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # make a prediction on the ROI, then lookup the class
        preds = classifier.predict(roi)[0]
        label = class_labels[preds.argmax()]  
        #label = 'Chris'
        y = rect[1] - 10 if rect[1] - 10 > 10 else rect[1] + 10
        label_position = (rect[0], y)

        if label=='Really Engaged': brd0_count += 1
        elif label=='Doing OK': brd1_count += 1
        elif label == 'A Bit Bored': brd2_count += 1
        elif label == 'Checked Out': brd3_count += 1

        if args.record == 'y':
            cv2.putText(image, 'Recording', (600, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

        cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]),(0, 255, 0), 2)
        cv2.putText(image, label, (label_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(image, 'Really Engaged: ' + str(brd0_count), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, 'Doing OK:' + str(brd1_count), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, 'A Bit Bored: ' + str(brd2_count), (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, 'Checked Out: ' + str(brd3_count), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    else:
        cv2.putText(image, "No Face Found", (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,2, (0, 0, 255), 3)

    if args.record == 'y':
        writerC.write(image)
    cv2.imshow('All', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()

msg = 'Really Engaged: ' + str(brd0_count) + \
      '\nDoing OK:' + str(brd1_count) + \
      '\nA Bit Bored: ' + str(brd2_count) + \
      '\nChecked Out: ' + str(brd3_count)

with open(rep_filename, 'w+') as writefile:
    writefile.write(msg)

if args.messaging == 'y':
    mqtt_client.publish(LOCAL_MQTT_TOPIC, msg, qos=0, retain=False)