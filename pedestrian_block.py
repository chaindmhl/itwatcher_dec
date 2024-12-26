import os
import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Now import other modules and start TensorFlow code
import tracking.deepsort_tric.core.utils as utils
from tensorflow.python.saved_model import tag_constants
from tracking.deepsort_tric.core.config_tc import cfg
from PIL import Image
import cv2
import numpy as np


from tracking.deepsort_tric.helper.read_plate import YOLOv4Inference
from tracking.deepsort_tric.helper.light_state import get_current_light_state
from tracking.deepsort_tric.helper.traffic_light import overlay_traffic_light
from tracking.deepsort_tric.helper.detect_color import Detect_Color
from tracking.deepsort_tric.helper.detect_plate import Detect_Plate
# deep sort imports
from tracking.deepsort_tric.deep_sort import preprocessing, nn_matching
from tracking.deepsort_tric.deep_sort.detection import Detection
from tracking.deepsort_tric.deep_sort.tracker import Tracker
from tracking.deepsort_tric.tools import generate_detections as gdet
from collections import deque, defaultdict
import math
import tempfile
import time, queue, datetime
from collections import Counter, deque
from tracking.models import ViolationLog

yolo_inference = YOLOv4Inference()
detect_color = Detect_Color()
detect_plate = Detect_Plate()
stop_threads = False


class Pedestrian_Blocking():
    def __init__(self, file_counter_log_name, framework='tf', weights='./checkpoints/lpd_comb',
                size=416, tiny=False, model='yolov4', video='./data/videos/cam0.mp4', outputfile=None,
                output=None, output_format='XVID', iou=0.45, score=0.5,
                dont_show=False, info=False,
                detection_line=(0.5,0), frame_queue = queue.Queue(maxsize=100), processed_queue = queue.Queue(maxsize=100), processing_time=0):
    
    
        self._file_counter_log_name = file_counter_log_name
        self._framework = framework
        self._weights = weights
        self._size = size
        self._tiny = tiny
        self._model = model
        self._video = video
        self._output = output
        self._output_format = output_format
        self._iou = iou
        self._score = score
        self._dont_show = dont_show
        self._info = info
        self._detect_line_position = detection_line[0]
        self._detect_line_angle = detection_line[1]
        self._queue = frame_queue
        self._processedqueue = processed_queue
        self._time = processing_time
        self._stop_threads = False 


    def _intersect(self, A, B, C, D):
        return self._ccw(A,C,D) != self._ccw(B, C, D) and self._ccw(A,B,C) != self._ccw(A,B,D)


    def _ccw(self, A,B,C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def _vector_angle(self, midpoint, previous_midpoint):
        x = midpoint[0] - previous_midpoint[0]
        y = midpoint[1] - previous_midpoint[1]
        return math.degrees(math.atan2(y, x))
    
    def _bbox_within_roi(self, bbox, roi_vertices):
        # Extract the bounding box coordinates
        xmin, ymin, xmax, ymax = map(int, bbox)

        # Define the corners and center points of the bounding box
        points_to_check = [
            (xmin, ymin),          # Top-left corner
            (xmax, ymin),          # Top-right corner
            (xmin, ymax),          # Bottom-left corner
            (xmax, ymax),          # Bottom-right corner
            ((xmin + xmax) // 2, (ymin + ymax) // 2),  # Center point
            ((xmin + xmax) // 2, ymin),                # Top-center
            ((xmin + xmax) // 2, ymax),                # Bottom-center
            (xmin, (ymin + ymax) // 2),                # Left-center
            (xmax, (ymin + ymax) // 2)                 # Right-center
        ]

        # Convert ROI vertices to a NumPy array
        roi_polygon = np.array(roi_vertices, dtype=np.int32)

        # Check if any of the points are within the polygonal ROI
        for point in points_to_check:
            if cv2.pointPolygonTest(roi_polygon, point, False) >= 0:
                return True

        return False

    
    def _process_frame(self,frame, input_size, infer, encoder, tracker, memory, violators = {}, plate_num_dict = {}, timer_start_time = {}, vehicle_colors ={}, plate_nums = {}, nms_max_overlap=0.1):
        memory = {}
        class_counter = Counter()

        batch_size = 1
        frame_size = frame.shape[:2]

        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(dtype=np.float32)

        # Repeat along the batch dimension to create a batch of desired size
        batch_data = np.repeat(image_data, batch_size, axis=0)

        # Convert to TensorFlow constant
        batch_data = tf.constant(batch_data, dtype=tf.float32)
        pred_bbox = infer(batch_data)
        for _, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self._iou,
            score_threshold=self._score
        )

        # Convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # Format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # Store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # Read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # By default allow all classes in .names file
        allowed_classes = list(class_names.values())

        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)

        # Delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # Encode YOLO detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        # Run non-maxima suppression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

                # Call the tracker
        tracker.predict()
        tracker.update(detections)

        #For Intersection
        roi_vertices = [
                (0, 393),      # Top-left (0, 741)
                (1591, 298),  # Top-right (1618, 602)
                (2425, 559),  # Bottom-right
                (2, 813)               # Bottom-left
            ]

        # Convert the vertices to a NumPy array of shape (vertices_count, 1, 2)
        roi_vertices_np = np.array(roi_vertices, dtype=np.int32)
        roi_vertices_np = roi_vertices_np.reshape((-1, 1, 2))
        # Ensure a global dictionary to store violation information for each track_id
        violation_info = {}

        # Draw the polygonal ROI using polylines
        cv2.polylines(frame, [roi_vertices_np], isClosed=True, color=(0, 0, 255), thickness=2)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            class_name = track.get_class()

            midpoint = track.tlbr_midpoint(bbox)
            origin_midpoint = (midpoint[0], frame.shape[0] - midpoint[1])

            track_id = str(track.track_id)

            if track_id not in memory:
                memory[track_id] = deque(maxlen=2)

            memory[track_id].append(midpoint)
            previous_midpoint = memory[track_id][0]

            # Calculate bbox points
            bbox_top_left = (int(bbox[0]), int(bbox[1]))
            bbox_bottom_right = (int(bbox[2]), int(bbox[3]))

            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            # Initialize color and elapsed_seconds with default values
            color = (0, 255, 0)  # Default to green color for non-violators
            elapsed_seconds = 0  # Default value for elapsed time

            if self._bbox_within_roi(bbox, roi_vertices):
                processing_start = time.time()  # Start processing timer
                # Initialize default values
                color = (0, 255, 0)  # Default color
                elapsed_seconds = 0  # Default elapsed time

                # Start the timer when entering the ROI
                if track_id not in timer_start_time:
                    timer_start_time[track_id] = datetime.datetime.now()

                # Calculate elapsed time
                elapsed_time = datetime.datetime.now() - timer_start_time[track_id]
                elapsed_seconds = elapsed_time.total_seconds()

                # Mark as violator after 5 seconds
                if elapsed_seconds >= 5 and track_id not in violators:
                    setattr(track, 'violated_blocking', True)
                    violators[track_id] = True
                    color = (0, 0, 255)  # Change to red for violators
                    
                    try:
                        # Log the violation once
                        xmin, ymin, xmax, ymax = map(int, bbox)
                        xmin = max(0, int(xmin))
                        ymin = max(0, int(ymin))
                        xmax = min(frame.shape[1] - 1, int(xmax))
                        ymax = min(frame.shape[0] - 1, int(ymax))

                        color_start = time.time()
                        # Crop vehicle image
                        vehicle_img = frame[ymin:ymax, xmin:xmax]
                        veh_img = cv2.cvtColor(vehicle_img, cv2.COLOR_RGB2BGR)
                        frame_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        veh_resized = cv2.resize(veh_img, (600, 300), interpolation=cv2.INTER_LANCZOS4)
                        veh_name = f"{track.track_id}.jpg"

                        # Perform color detection
                        color_pred = detect_color.infer(vehicle_img, track.track_id)
                        clr = color_pred.get("detected_class", "Unknown")  # Default to "Unknown" if not detected
                        color_end = time.time()

                        if clr is not None:
                            # Store detected color for the track ID
                            vehicle_colors[track.track_id] = clr

                        plate_start = time.time()
                        # Perform plate detection
                        plate_pred = detect_plate.infer_image(veh_resized)
                        plate_disp = plate_pred.get("detected_class", "Unknown")  # Default to "Unknown" if not detected
                        plate_end = time.time()

                        recognition_start = time.time()
                        cropped_plate = plate_pred.get("cropped_plate", np.zeros((1, 1, 3), dtype=np.uint8))  # Default to an empty image
                        plate_resized = cv2.resize(cropped_plate, (2000, 600), interpolation=cv2.INTER_LANCZOS4)
                        pred = yolo_inference.infer_image_only_thresh(plate_resized)
                        plate_disp = "".join(pred.get("detected_classes", ["Unknown"]))  # Default to "Unknown" if not detected
                        recognition_end = time.time()
                        if plate_disp is not None:
                            # Store detected color for the track ID
                            plate_nums[track.track_id] = plate_disp
        
                        # Save to the database if plate number is not already logged
                        if plate_disp not in plate_num_dict:
                            plate_num_dict[plate_disp] = time.time()

                            db_save_start = time.time()
                            # Create and save violation log
                            vio_log = ViolationLog.objects.create(
                                video_file=self._video,
                                vehicle_type = class_name,
                                violation='Blocking Pedestrian Lane',
                                plate_number=plate_disp,
                                vehicle_color=clr,
                            )
                            db_save_end = time.time()
                            
                            # Output the timings for each step
                            print(f"Track ID: {track_id} - {class_name} processing times:")
                            print(f"Color Detection: {color_end - color_start:.4f}s")
                            print(f"Plate Detection: {plate_end - plate_start:.4f}s")
                            print(f"Plate Recognition: {recognition_end - recognition_start:.4f}s")
                            print(f"Database Save: {db_save_end - db_save_start:.4f}s")
                            print(f"Total processing: {time.time() - processing_start:.4f}s")
                            # Save vehicle, plate, and frame images
                            vehicle_img_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                            Image.fromarray(cv2.cvtColor(vehicle_img, cv2.COLOR_RGB2BGR)).save(vehicle_img_temp.name)
                            vehicle_img_temp.close()

                            plate_img_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                            Image.fromarray(cropped_plate).save(plate_img_temp.name)
                            plate_img_temp.close()

                            frame_img_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                            Image.fromarray(frame_img).save(frame_img_temp.name)
                            frame_img_temp.close()

                            vio_log.vehicle_image.save(veh_name, open(vehicle_img_temp.name, 'rb'))
                            vio_log.plate_image.save(plate_disp + ".jpg", open(plate_img_temp.name, 'rb'))
                            vio_log.frame_image.save(plate_disp + ".jpg", open(frame_img_temp.name, 'rb'))

                            # Remove temporary files
                            os.unlink(vehicle_img_temp.name)
                            os.unlink(plate_img_temp.name)
                            os.unlink(frame_img_temp.name)

                    except cv2.error as e:
                        continue
                
                elif hasattr(track, 'violated_blocking') and track.violated_blocking:

                    # Ensure clr and plate_disp are defined
                    clr = vehicle_colors.get(track.track_id, "Unknown")
                    plate_disp = plate_nums.get(track.track_id, "Unknown")
                    
                    # Display violation status even after crossing
                    cv2.rectangle(frame, bbox_top_left, bbox_bottom_right, (0, 0, 255), 2)
                    cv2.putText(frame, f"Vehicle Type:{class_name}", (bbox_top_left[0], bbox_top_left[1] - 60), 0,
                                0.7e-3 * frame.shape[0], (255, 125, 125), 2)
                    cv2.putText(frame, f"Color:{clr}", (bbox_top_left[0], bbox_top_left[1]- 35), 0,
                                0.7e-3 * frame.shape[0], (255, 125, 125), 2)
                    cv2.putText(frame, f"Plate Number: {plate_disp}", (bbox_top_left[0], bbox_top_left[1]- 10), 0,
                                0.7e-3 * frame.shape[0], (255, 125, 125), 2)
                else:

                    # Display normal bounding box
                    cv2.rectangle(frame, bbox_top_left, bbox_bottom_right, (0, 255, 0), 2)

                            
        # This needs to be larger than the number of tracked objects in the frame.
        if len(memory) > 50:
            del memory[list(memory)[0]]

        result = np.asarray(frame)
        return result

    
    def producer(self):

        global stop_threads
        frame_count = 0
        skip_frames = 1

        cap = cv2.VideoCapture(self._video)
        if not cap.isOpened():
            # print("Error: Unable to open the video stream.")
            return
        
        while not stop_threads:
            ret, frame = cap.read()
            # print("reading video...")
            if not ret:
                # print("Failed to retrieve frame. Pausing...")
                stop_threads = False
                continue
            frame_count +=1

            if frame_count % skip_frames == 0: 
                try:
                    self._queue.put(frame, timeout=1)

                except queue.Full:
                    
                    time.sleep(1)
                    continue
                    

        cap.release()

    def consumer(self):
        global stop_threads
        input_size = self._size
        total_processing_time = 0
        num_frames_processed = 0
        frame_count = 0 

        # Load configuration for object detector
        saved_model_loaded = tf.saved_model.load(self._weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
        model_filename = '/home/icebox/itwatcher_api/tracking/deepsort_tric/model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        max_cosine_distance = 0.4
        nn_budget = None
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)
        memory = {}
        

        while not stop_threads:
            try:
                frame = self._queue.get(timeout=1)
                
            except queue.Empty:
                continue
            
            start_time = time.time()

            result = self._process_frame(frame, input_size, infer, encoder, tracker, memory)

            end_time = time.time()
            processing_time = end_time - start_time
            total_processing_time += processing_time
            num_frames_processed += 1

            if result is not None and len(result) > 0:
                self._processedqueue.put(result)

        # Calculate average processing time
        average_processing_time = 0
        if num_frames_processed > 0:
            average_processing_time = total_processing_time / num_frames_processed
            print(f"Average processing time: {average_processing_time:.3f} seconds") 

        self._time = average_processing_time  # Set the attribute
        print(self._time)
        return average_processing_time  # Return average processing time
        
    def retrieve_processed_frames(self):
            processed_frames = []
            while not self._processedqueue.empty():
                processed_frames.append(self._processedqueue.get())
            return processed_frames
        
    def stop(self):
        self._stop_threads = True
    
session.close()