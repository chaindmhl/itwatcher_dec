import cv2
import numpy as np
import os

class Detect_Color:
    def __init__(self, weights_path="/home/icebox/itwatcher_api/darknet/clr/color.weights",
                 config_path="/home/icebox/itwatcher_api/darknet/clr/yolov4-custom.cfg",
                 class_names_path="/home/icebox/itwatcher_api/darknet/clr/color.names",
                 confidence_threshold=0.5, nms_threshold=0.4):
        self.weights_path = weights_path
        self.config_path = config_path
        self.class_names_path = class_names_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        with open(self.class_names_path, 'r') as f:
            self.classes = f.read().strip().split('\n')

    def infer_image(self, img):
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        outputs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

        detected_classes = []
        if indexes is not None and len(indexes) > 0:
            for i in indexes.flatten():  # Updated to handle indexes properly
                label = str(self.classes[class_ids[i]])
                detected_classes.append((label, confidences[i]))  # Include confidence for more precise handling

        if detected_classes:
            # Sort detected classes by confidence score in descending order and return the one with highest confidence
            detected_classes.sort(key=lambda x: x[1], reverse=True)
            best_class = detected_classes[0][0]
        else:
            best_class = None

        return img, best_class

    def infer(self, img, track_id):
        # Perform inference and get the annotated image and detected classes with x-coordinates
        annotated_image, best_detected_class = self.infer_image(img)

        # Return the best detected class and the corresponding track ID
        if best_detected_class:
            return {"detected_class": best_detected_class, "track_id": track_id}

        return {"detected_class": None, "track_id": None}  # Return None if no classes are detected
