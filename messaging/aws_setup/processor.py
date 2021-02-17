import paho.mqtt.client as mqtt
from datetime import datetime

# local info
LOCAL_MQTT_HOST = "aws_broker"
LOCAL_MQTT_PORT = 1883
LOCAL_MQTT_TOPIC = "engagement"
PATH = "/usr/src/app/engagement/s3/"


# local callback function
def on_connect_local(client, userdata, flags, rc):
    print("image processor connected to aws broker with rc: " + str(rc))
    client.subscribe(LOCAL_MQTT_TOPIC)


def on_message(client, userdata, msg):
    try:
        print("\nreport to process received")
        rep = msg.payload
        ts = datetime.utcnow().strftime('%Y%m%d-%H%M%S.%f')
        filename = PATH + 'engagement_report_' + ts + '.txt'
        print("Saving", filename)

        with open(filename, "w+") as outfile:
            outfile.write(rep)

    except:
        print("Unexpected error:", sys.exc_info()[0])

local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_mqttclient.on_message = on_message

# go into a loop
