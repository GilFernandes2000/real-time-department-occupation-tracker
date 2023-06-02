import torch
import cv2
import numpy as np
from collections import Counter
import paho.mqtt.client as mqtt
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from multiprocessing import Value
import threading
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

# The callback for when a PUBLISH message is received from the server
def on_message(client, userdata, message):
    message = message.payload.decode()
    if message == 'Enter':
        people_inside.value += 1
        update_people_inside(people_inside.value)
        print(f'Update from Jetson: Person entered the building. People inside: {people_inside.value}')
    elif message == 'Leave': 
        people_inside.value -= 1
        people_inside.value = max(0, people_inside.value)
        update_people_inside(people_inside.value)
        print(f'Update from Jetson: Person left the building. People inside: {people_inside.value}')

# Create an MQTT client and subscribe to topic
client = mqtt.Client()
client.on_message = on_message
client.connect(MQTT_SERVER)
client.subscribe(MQTT_PATH)

client.loop_start()

@app.route('/')
def home():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(target=main)
    return render_template('dashboard.html')

def update_people_inside(new_value):
    with people_inside.get_lock():
        people_inside.value = new_value
        socketio.emit('people_inside_update', {'people_inside': people_inside.value})


# Load model
model = torch.hub.load('GilFernandes2000/yolov5', 'yolov5s')

def xyxy2bbox(xyxy):
    return [int(coord) for coord in xyxy]

def calculate_direction(prev_centroid, curr_centroid):
    y_diff = curr_centroid[1] - prev_centroid[1]
    return 'Down' if y_diff > 0 else 'Up' if y_diff < 0 else 'Stationary'

def process_frame(frame):
    results = model(frame)
    return results

def track_objects(results, frame):
    global id_counter, people_inside
    line_y = frame.shape[0] // 2
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 2) # Line that server as entrance/exit

    for *xyxy, conf, cls in results.xyxy[0]:
        cls_name = results.names[int(cls)]
        if cls_name != 'person':
            continue
        bbox = xyxy2bbox(xyxy)
        centroid = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

        for obj_id, obj_info in objects.items():
            if np.linalg.norm(np.array(obj_info['centroid']) - np.array(centroid)) < 50:
                break
        else:
            obj_id = id_counter
            id_counter += 1
            objects[obj_id] = {'centroid': centroid, 'cls_name': cls_name, 'directions': [], 'side': 'Up' if centroid[1] < line_y else 'Down', 'state': 'Unknown'}

        if objects[obj_id]['directions'] and objects[obj_id]['directions'][-1] != 'Stationary':
            old_people_inside = people_inside.value
            
            # A Persons enters the building
            if objects[obj_id]['directions'][-1] == 'Down' and centroid[1] > line_y and objects[obj_id]['side'] == 'Up' and objects[obj_id]['state'] != 'Inside':
                objects[obj_id]['side'] = 'Down'
                objects[obj_id]['state'] = 'Inside'
                print(f'Person {obj_id} entered the building')
                people_inside.value += 1
            
            # A Person leaves the building
            elif objects[obj_id]['directions'][-1] == 'Up' and centroid[1] < line_y and objects[obj_id]['side'] == 'Down' and objects[obj_id]['state'] != 'Outside':
                objects[obj_id]['side'] = 'Up'
                objects[obj_id]['state'] = 'Outside'
                print(f'Person {obj_id} exited the building')
                people_inside.value -= 1
                people_inside.value = max(0, people_inside.value)
            
            if old_people_inside != people_inside.value:
                print(f'Estimated number of people inside the building: {people_inside.value}')
                
                update_people_inside(people_inside.value)

        draw_objects(frame, bbox, centroid, cls_name, obj_id)
        update_object_direction(obj_id, centroid)

    return frame

def draw_objects(frame, bbox, centroid, cls_name, obj_id):
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
    cv2.putText(frame, f'{cls_name}_{obj_id}', (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

def update_object_direction(obj_id, centroid):
    if 'directions' in objects[obj_id]:
        direction = calculate_direction(objects[obj_id]['centroid'], centroid)
        if direction != 'Stationary':
            objects[obj_id]['directions'].append(direction)
        if len(objects[obj_id]['directions']) == num_frames:
            direction_counts = Counter(objects[obj_id]['directions'])
            overall_direction = direction_counts.most_common(1)[0][0]
            if 'overall_direction' in objects[obj_id] and objects[obj_id]['overall_direction'] != overall_direction:
                objects[obj_id]['overall_direction'] = overall_direction
            objects[obj_id]['directions'] = []
        objects[obj_id]['centroid'] = centroid


def main():
    try:
        cap = cv2.VideoCapture("peoplewalking.mp4")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = process_frame(frame)
            frame = track_objects(results, frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
    except KeyboardInterrupt:
        print("Received interrupt, stopping server...")
    finally:
        with open('people_inside.txt', 'w') as file:
            file.write(str(people_inside))
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000)