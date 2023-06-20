import torch
import cv2
import numpy as np
from collections import Counter
import threading
import paho.mqtt.client as mqtt
from ultralytics import YOLO
import time

MQTT_SERVER = "10.1.1.20"
MQTT_PATH = "counter"

client = mqtt.Client()
client.connect(MQTT_SERVER)

# Define a dictionary to store information about each object
objects = {}

# ID counter
id_counter = 1

# Number of frames to collect before estimating direction
num_frames = 3

last_frame = None

sig = False

people_inside = 0

# Load model
model = YOLO('yolov8n.pt')


def xyxy2bbox(xyxy):
    """_summary_
    Convert xyxy format to bbox format
    
    Args:
        xyxy: xyxy format
    Returns: bbox format
    """
    return [int(coord) for coord in xyxy]


def calculate_direction(prev_centroid, curr_centroid):
    """_summary_
    Calculates the direction of the object based on the previous and current centroid position
    
    Args:
        prev_centroid: centroid position in the previous frame
        curr_centroid: centroid position in the current frame
    Returns: direction of the object
    """
    y_diff = curr_centroid[1] - prev_centroid[1]
    return 'Down' if y_diff > 0 else 'Up' if y_diff < 0 else 'Stationary'


def process_frame(frame):
    """_summary_
    Process the frame with the YOLOv8 model
    
    Args:
        frame: frame to be processed
        
    Returns: results of the model
    """
    results = model(frame)
    return results


def track_objects(results, frame):
    """_summary_
    Process the results of the model and track the objects
    Draw the bounding boxes and the centroids of the objects
    Estimate the direction of the objects
    Update the people_inside counter
    
    Args:
        results: results of the model
        frame: frame to be processed
        
    Returns: frame with the objects tracked
    """
    
    global id_counter, people_inside
    line_y = frame.shape[0] // 2
    line_y = line_y + 75
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 2)
    r = results[0].cpu().numpy()
    boxes = r.boxes.xyxy
    confs = r.boxes.conf
    clss = r.boxes.cls

    for i in range(len(boxes)):
        cls_name = r.names[int(clss[i])]
        if cls_name != 'person':
            continue
        bbox = xyxy2bbox(boxes[i])
        centroid = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

        for obj_id, obj_info in objects.items():
            if np.linalg.norm(np.array(obj_info['centroid']) - np.array(centroid)) < 50:
                break
        else:
            obj_id = id_counter
            id_counter += 1
            objects[obj_id] = {'centroid': centroid, 'cls_name': cls_name, 'directions': [], 'side': 'Up' if centroid[1] < line_y else 'Down', 'state': 'Unknown'}

        if objects[obj_id]['directions'] and objects[obj_id]['directions'][-1] != 'Stationary':
            
            if objects[obj_id]['directions'][-1] == 'Down' and centroid[1] > line_y and objects[obj_id]['side'] == 'Up' and objects[obj_id]['state'] != 'Inside':
                objects[obj_id]['side'] = 'Down'
                objects[obj_id]['state'] = 'Inside'
                #client.publish(MQTT_PATH, 'Enter')
                print(f'Person {obj_id} entered the building')
                people_inside += 1
            elif objects[obj_id]['directions'][-1] == 'Up' and centroid[1] < line_y and objects[obj_id]['side'] == 'Down' and objects[obj_id]['state'] != 'Outside':
                objects[obj_id]['side'] = 'Up'
                objects[obj_id]['state'] = 'Outside'
                #client.publish(MQTT_PATH, 'Leave')
                print(f'Person {obj_id} exited the building')
                people_inside -= 1
                people_inside = max(0, people_inside)


        draw_objects(frame, bbox, centroid, cls_name, obj_id)
        update_object_direction(obj_id, centroid)

    return frame


def draw_objects(frame, bbox, centroid, cls_name, obj_id):
    """_summary_
    Draw the bounding boxes and the centroids of the objects
    
    
    Args:
        frame: frame to be processed
        bbox: bounding box of the object
        centroid: centroid of the object
        cls_name: class name of the object
        obj_id: id of the object
    """
    
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
    cv2.putText(frame, f'{cls_name}_{obj_id}', (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)


def update_object_direction(obj_id, centroid):
    """_summary_
    Update the direction of the object based on the previous and current centroid position
    
    Args:
        obj_id: id of the object
        centroid: centroid of the object
    """
    
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


def thread_get_most_recent_frame():
    """_summary_
    Separate thread to get the most recent frame of the video
    While the main thread is processing the previous frame, this thread is getting the next frame
    """
    
    global last_frame
    global sig
    video_path = "rtsp://admin:napit2023@192.168.1.30:554/cam/realmonitor?channel=1&subtype=1"
    cap = cv2.VideoCapture(video_path)
    while not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
    sig = True

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        success, frame = cap.read()
        if success:
           last_frame = frame
        else:
            break
    cap.release()
    
def main():
    p_frames = threading.Thread(target=thread_get_most_recent_frame)
    p_frames.start()

    while not sig:
        time.sleep(1)
    
    while True:
        frame_data = last_frame

        results = process_frame(frame_data)
        frame_data = track_objects(results, frame_data)
        cv2.imshow('frame', frame_data)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
