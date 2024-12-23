import cv2
import numpy as np
import os


yolo_cfg_path = 'C:/Users/admin/OneDrive/Desktop/MINI_PROJECT/CODE1/yolov3.cfg'
yolo_weights_path = 'C:/Users/admin/OneDrive/Desktop/MINI_PROJECT/CODE1/yolov3.weights'


if not os.path.isfile(yolo_cfg_path):
    raise FileNotFoundError(f"YOLO configuration file not found: {yolo_cfg_path}")
if not os.path.isfile(yolo_weights_path):
    raise FileNotFoundError(f"YOLO weights file not found: {yolo_weights_path}")

net = cv2.dnn.readNet(yolo_weights_path, yolo_cfg_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


cap = cv2.VideoCapture('C:/Users/admin/OneDrive/Desktop/MINI_PROJECT/CODE1/inputs/v3.mp4')


output_width = 700  
output_height = 500  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    
    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = center_x - w // 2
                y = center_y - h // 2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

   
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = "Accident" 
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)  
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)  

    
    resized_frame = cv2.resize(frame, (output_width, output_height))
    cv2.imshow('Frame', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
