import cv2
from ultralytics import YOLO
import numpy as np
from collections import Counter
import paho.mqtt.client as mqtt
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from multiprocessing import Value
from threading import Lock

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')
thread = None
thread_lock = Lock()

MQTT_SERVER = "10.1.1.20" # B.A.T.M.A.N. ip of the master
MQTT_PATH = "counter" # Topic name on the master

client = mqtt.Client()
client.connect(MQTT_SERVER)

# Dictionary to store information about each object
objects = {}

# ID counter
id_counter = 1

# Number of frames to collect before estimating direction
num_frames = 10

# Counter for people inside
people_inside = Value('i', 0)

def on_message(message):
    """_summary_
    This function is called when a message is received from the Jetson Worker.
    It adds the value contained in the message to the people_inside counter.
    Args:
        message: message received from the Jetson Worker
    """
    
    message = message.payload.decode()
    if message == 'Enter':
        with people_inside.get_lock():
            people_inside.value += 1
            socketio.emit('people_inside_update', {'people_inside': people_inside.value})
            print(f'Update from Jetson: Person entered the building. People inside: {people_inside.value}')
    elif message == 'Leave': 
        with people_inside.get_lock():
            people_inside.value -= 1
            people_inside.value = max(0, people_inside.value)
            socketio.emit('people_inside_update', {'people_inside': people_inside.value})
            print(f'Update from Jetson: Person left the building. People inside: {people_inside.value}')

# Create an MQTT client and subscribe to topic
client = mqtt.Client()
client.on_message = on_message
client.connect(MQTT_SERVER)
client.subscribe(MQTT_PATH)

client.loop_start()

@app.route('/')
def home():
    return render_template('dashboard.html', people_inside=people_inside.value)

if __name__ == "__main__":
    socketio.run(app, host='192.168.1.20', port=5000)
