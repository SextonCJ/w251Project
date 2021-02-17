import paho.mqtt.client as mqtt

# remote 
REMOTE_MQTT_HOST = "54.69.37.72"
REMOTE_MQTT_PORT = 1883
REMOTE_MQTT_TOPIC = "engagement"

# local 
LOCAL_MQTT_HOST = "mosquitto"
LOCAL_MQTT_PORT = 1883
LOCAL_MQTT_TOPIC = "jetson/report"


# remote callback function
def on_publish_remote(client, userdata, result):
    print("data published to remote server \n")
    pass


# local callback function
def on_connect_local(client, userdata, flags, rc):
    print("connected to local broker with rc: " + str(rc))
    client.subscribe(LOCAL_MQTT_TOPIC)


# on message receive function
def on_message(client, userdata, msg):
    try:
        print("Message received!")
        remote_mqtt_client.publish(REMOTE_MQTT_TOPIC, payload=msg.payload, qos=0, retain=False)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        

# create remote mqtt client
remote_mqtt_client = mqtt.Client()
remote_mqtt_client.on_publish = on_publish_remote
remote_mqtt_client.connect(REMOTE_MQTT_HOST, REMOTE_MQTT_PORT)

# create local mqtt client
local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_mqttclient.on_message = on_message


# go into a loop
local_mqttclient.loop_forever()
