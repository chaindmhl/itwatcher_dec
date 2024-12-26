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
from tracking.models import PlateLog

yolo_inference = YOLOv4Inference()
detect_plate = Detect_Plate()
stop_threads = False



class Plate_Recognition_comb():
    def __init__(self, file_counter_log_name, framework='tf', weights='./checkpoints/lpd_comb',
                size=416, tiny=False, model='yolov4', video='./data/videos/cam0.mp4', outputfile=None,
                output=None, output_format='XVID', iou=0.45, score=0.5,
                dont_show=False, info=False, allowed_classes=None,
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
        self._allowed_classes = allowed_classes if allowed_classes else []  # Default to empty list if not provided
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
    
    def _plate_within_roi(self, bbox, roi_vertices):
        # Calculate the center of the bounding box
        xmin, ymin, xmax, ymax = map(int, bbox)
        bbox_center = ((xmin + xmax) // 2, (ymin + ymax) // 2)
        
        # Check if the bbox_center is within the polygonal ROI
        roi_polygon = np.array(roi_vertices, dtype=np.int32)
        is_within_roi = cv2.pointPolygonTest(roi_polygon, bbox_center, False) >= 0
        
        return is_within_roi
    
    def _process_frame(self, frame, input_size, infer, encoder, tracker, memory, already_save={}, plate_display={}, plate_nums={}, plate_num_dict={}, nms_max_overlap=0.1):
        # Preprocess frame and perform inference
        batch_data, original_h, original_w = self._preprocess_frame(frame, input_size)
        pred_bbox = infer(batch_data)

        # Extract bounding boxes, scores, and classes
        bboxes, scores, classes, num_objects = self._extract_detections(pred_bbox, original_h, original_w)

        # Filter detections and encode features
        detections = self._filter_and_encode_detections(frame, bboxes, scores, classes, encoder)

        # Run non-maxima suppression
        detections = self._apply_nms(detections, nms_max_overlap)

        # Track objects
        self._track_objects(tracker, detections)

        # Define lines for intersection checks
        lines = self._define_lines(frame)

        # Draw lines on the frame
        # for line in lines:
        #     cv2.line(frame, line[0], line[1], (0, 255, 0), 2)  # Draw each line in green with thickness 2

        # Process tracked objects and check for intersections
        self._process_tracked_objects(frame, tracker, memory, already_save, plate_display, plate_nums, plate_num_dict, lines)

        # Clean up memory if needed
        self._cleanup_memory(memory)

        return np.asarray(frame)

    def _preprocess_frame(self, frame, input_size):
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(dtype=np.float32)
        batch_data = tf.constant(np.repeat(image_data, 1, axis=0), dtype=tf.float32)
        return batch_data, frame.shape[0], frame.shape[1]

    def _extract_detections(self, pred_bbox, original_h, original_w):
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

        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0][0:int(num_objects)]
        scores = scores.numpy()[0][0:int(num_objects)]
        classes = classes.numpy()[0][0:int(num_objects)]

        bboxes = utils.format_boxes(bboxes, original_h, original_w)
        return bboxes, scores, classes, num_objects

    def _filter_and_encode_detections(self, frame, bboxes, scores, classes, encoder):
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        # allowed_classes = list(class_names.values())
        allowed_classes = self._allowed_classes  # Use the instance's allowed classes
        names = []
        deleted_indx = []

        for i in range(len(classes)):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)

        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]
        return detections

    def _apply_nms(self, detections, nms_max_overlap):
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        return [detections[i] for i in indices]

    def _track_objects(self, tracker, detections):
        tracker.predict()
        tracker.update(detections)

    def _define_lines(self, frame):
        lines = []

        # Original vertical lines
        # Vertical line 1 (center)
        x1 = int(frame.shape[1] / 2)
        y1 = 0
        x2 = int(frame.shape[1] / 2)
        y2 = int(frame.shape[0])
        lines.append([(x1, y1), (x2, y2)])  # line1

        # Vertical line 2 (1/4)
        x3 = int(frame.shape[1] / 4)
        y3 = 0
        x4 = int(frame.shape[1] / 4)
        y4 = int(frame.shape[0])
        lines.append([(x3, y3), (x4, y4)])  # line1a

        # Vertical line 3 (3/4)
        x5 = int(3 * frame.shape[1] / 4)
        y5 = 0
        x6 = int(3 * frame.shape[1] / 4)
        y6 = int(frame.shape[0])
        lines.append([(x5, y5), (x6, y6)])  # line1b

        # Original horizontal lines
        # Horizontal line 1
        xa = 0
        ya = int((frame.shape[0] / 4) + 150)
        xb = int(frame.shape[1])
        yb = int((frame.shape[0] / 4) + 150)
        lines.append([(xa, ya), (xb, yb)])  # line2

        # Horizontal line 2 (middle)
        xc = 0
        yc = int(frame.shape[0] / 2)
        xd = int(frame.shape[1])
        yd = int(frame.shape[0] / 2)
        lines.append([(xc, yc), (xd, yd)])  # line3

        # Horizontal line 3 (middle + 200)
        xe = 0
        ye = int((frame.shape[0] / 2) + 200)
        xf = int(frame.shape[1])
        yf = int((frame.shape[0] / 2) + 200)
        lines.append([(xe, ye), (xf, yf)])  # line4

        # Horizontal line 4 (middle + 400)
        xg = 0
        yg = int((frame.shape[0] / 2) + 400)
        xh = int(frame.shape[1])
        yh = int((frame.shape[0] / 2) + 400)
        lines.append([(xg, yg), (xh, yh)])  # line5

        # Horizontal line 5 (middle + 600)
        xi = 0
        yi = int((frame.shape[0] / 2) + 600)
        xj = int(frame.shape[1])
        yj = int((frame.shape[0] / 2) + 600)
        lines.append([(xi, yi), (xj, yj)])  # line6

        return lines


    def _process_tracked_objects(self, frame, tracker, memory, already_save, plate_display, plate_nums, plate_num_dict, lines):
        for track in tracker.tracks:
            # Skip unconfirmed tracks or those that have not been updated recently
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            # Get bounding box and class name
            bbox = track.to_tlbr()
            class_name = track.get_class()

            # Calculate the midpoint of the bounding box
            midpoint = track.tlbr_midpoint(bbox)
            origin_midpoint = (midpoint[0], frame.shape[0] - midpoint[1])

            # Store current midpoint in memory with a maximum length of 2 for previous midpoints
            if track.track_id not in memory:
                memory[track.track_id] = deque(maxlen=2)
            
            # Add the current midpoint to the memory
            memory[track.track_id].append(midpoint)
            previous_midpoint = memory[track.track_id][0]

            origin_previous_midpoint = (previous_midpoint[0], frame.shape[0] - previous_midpoint[1])
            # cv2.line(frame, midpoint, previous_midpoint, (0, 255, 0), 2)            

            # Draw bounding box
            plate_disp = plate_nums.get(track.track_id, " ")
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
            cv2.putText(frame, f"Plate Number: {plate_disp}", (int(bbox[0]), int(bbox[1]) - 10), 0, 0.7e-3 * frame.shape[0], (255, 125, 125), 2)

                   
            previous_midpoint = memory[track.track_id][0]  # Get the previous midpoint
            cv2.line(frame, (int(midpoint[0]), int(midpoint[1])), (int(previous_midpoint[0]), int(previous_midpoint[1])), (0, 255, 0), 2)  # Green line

            if track.track_id not in already_save:
                already_save[track.track_id] = False
            if track.track_id not in plate_display:
                plate_display[track.track_id] = False

            # Check for intersections
            if self._check_intersection(midpoint, memory[track.track_id][0], lines) and not plate_display[track.track_id] and not already_save[track.track_id]:
                self._process_vehicle(frame, track, bbox, plate_nums, plate_num_dict, already_save, midpoint, previous_midpoint, lines)


    def _update_memory(self, memory, track_id, midpoint):
        if track_id not in memory:
            memory[track_id] = deque(maxlen=2)
        memory[track_id].append(midpoint)

    def _check_intersection(self, midpoint, previous_midpoint, lines):
        intersects = any(self._intersect(midpoint, previous_midpoint, line[0], line[1]) for line in lines)
        # print(f"Intersection Check: {intersects}")
        return intersects

    def _process_vehicle(self, frame, track, bbox, plate_nums, plate_num_dict, already_save, midpoint, previous_midpoint, lines):
        # Check if the object intersects with any line and if it hasn't been already saved
        if not already_save.get(track.track_id, False) and self._check_intersection(midpoint, previous_midpoint, lines):
            try:
                vehicle_img = self._crop_vehicle(frame, bbox)
                plate_disp, cropped_plate = self._detect_and_recognize_plate(vehicle_img, track.track_id)

                if plate_disp is not None and plate_disp != "Unknown":
                    # print(f"Detected Plate: {plate_disp} for Track ID: {track.track_id}")
                    plate_nums[track.track_id] = plate_disp
                    self._save_plate_log(plate_disp, cropped_plate, frame, track.track_id, plate_num_dict)
                    already_save[track.track_id] = True
                else:
                    print(f"No plate detected for Track ID: {track.track_id}")

            except cv2.error as e:
                print(f"Error while processing vehicle: {e}")

    def _crop_vehicle(self, frame, bbox):
        xmin, ymin, xmax, ymax = map(int, bbox)
        allowance = 0
        xmin = max(0, int(xmin - allowance))
        ymin = max(0, int(ymin - allowance))
        xmax = min(frame.shape[1] - 1, int(xmax + allowance))
        ymax = min(frame.shape[0] - 1, int(ymax + allowance))
        cropped_vehicle = frame[ymin:ymax, xmin:xmax]
        # print(f"Cropped Vehicle Image Shape: {cropped_vehicle.shape}")
        return cropped_vehicle

    def _detect_and_recognize_plate(self, vehicle_img, track_id):
        veh_resized = cv2.resize(vehicle_img, (600, 300), interpolation=cv2.INTER_LANCZOS4)
        
        # Use YOLO inference for plate detection
        plate_pred = detect_plate.infer_image(veh_resized)  # Change for LPD
        # print(f"Plate Prediction Output: {plate_pred}")
        
        plate_disp = plate_pred.get("detected_class", "Unknown")
        
        # Ensure cropped_plate exists
        cropped_plate = plate_pred.get("cropped_plate", np.zeros((1, 1, 3), dtype=np.uint8))  # Default to an empty image
        plate_resized = cv2.resize(cropped_plate, (2000, 600), interpolation=cv2.INTER_LANCZOS4)
        
        # Use YOLO inference method for recognizing the plate
        pred = yolo_inference.infer_image_only_thresh(plate_resized)
        plate_disp = "".join(pred.get("detected_classes", ["Unknown"]))  # Default to "Unknown" if not detected

        # print(f"Detected Plate Display: {plate_disp}")
        return plate_disp, cropped_plate

    def _save_plate_log(self, plate_disp, cropped_plate, frame, track_id, plate_num_dict):
        image_name = plate_disp + ".jpg"
        current_timestamp = time.time()

        if plate_disp not in plate_num_dict:
            plate_num_dict[plate_disp] = current_timestamp

        plate_log = PlateLog.objects.create(
            video_file=self._video,
            plate_number=plate_disp
        )
        
        # print(f"Saving Plate Log: {plate_disp}")

        # Create temporary files for plate_img and frame
        cropped_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_RGB2BGR)
        plate_img_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        Image.fromarray(cropped_plate).save(plate_img_temp.name)
        plate_img_temp.close()

        frame_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_img_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        Image.fromarray(frame_img).save(frame_img_temp.name)
        frame_img_temp.close()

        # Save plate_image using ImageField
        plate_log.plate_image.save(image_name, open(plate_img_temp.name, 'rb'))
        # Save frame_image using ImageField
        plate_log.frame_image.save(image_name, open(frame_img_temp.name, 'rb'))

        # Clean up temporary files
        os.unlink(plate_img_temp.name)
        os.unlink(frame_img_temp.name)

        # print(f"Plate log saved for: {plate_disp}.")

    def _cleanup_memory(self, memory):
        for track_id in list(memory.keys()):
            if len(memory[track_id]) > 50:
                del memory[track_id]
    
    def producer(self):

        global stop_threads
        frame_count = 0
        skip_frames = 1

        cap = cv2.VideoCapture(self._video)
        if not cap.isOpened():
            # print("Error: Unable to open the video stream.")
            return
        
        while not self._stop_threads:
            ret, frame = cap.read()
 
            # If the end of the video is reached
            if not ret:
                print("Video finished or failed to retrieve frame.")
                break  # Exit the loop

            frame_count +=1

            if frame_count % skip_frames == 0: 
                try:
                    self._queue.put(frame, timeout=1)

                except queue.Full:
                    
                    time.sleep(1)
                    continue
                    

        cap.release()

        # Signal consumer that no more frames will be sent
        try:
            self._queue.put(None, timeout=1)  # Use None as a sentinel value
        except queue.Full:
            print("Warning: Queue was full, could not send sentinel value.")

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

        video_path = self._video

        # begin video capture
        try:
            vid = cv2.VideoCapture(int(video_path))
        except:
            vid = cv2.VideoCapture(video_path)

        out = None

        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*self._output_format)
        try:
            # Open VideoWriter
            out = cv2.VideoWriter(self._output, codec, fps, (width, height))

            
            while not self._stop_threads:
                try:
                    frame = self._queue.get(timeout=1)
                    
                except queue.Empty:
                    continue
                
                start_time = time.time()

                # if frame is None:  # Check for the sentinel value
                #     print("No more frames to process. Exiting consumer.")
                #     break

                # Call the new process frame function
                processed_frame = self._process_frame(frame, input_size, infer, encoder, tracker, memory)

                # Put the processed frame into the queue for output
                self._processedqueue.put(processed_frame)

                if processed_frame is not None and processed_frame.size > 0:
                    out.write(processed_frame)
                    # print(f"Writing frame {num_frames_processed}")
                else:
                    print("Warning: Attempted to write an invalid frame.")


                end_time = time.time()
                processing_time = end_time - start_time
                total_processing_time += processing_time
                num_frames_processed += 1
        
        finally:
            # Ensure the VideoWriter is released at the end
            if out is not None:
                out.release()
            vid.release()
                
        # Calculate average processing time
        average_processing_time = 0
        if num_frames_processed > 0:
            average_processing_time = total_processing_time / num_frames_processed
            # print(f"Average processing time: {average_processing_time:.3f} seconds") 

        self._time = average_processing_time  # Set the attribute
        # print(self._time)
        return average_processing_time  # Return average processing time
        
    def retrieve_processed_frames(self):
        processed_frames = []
        while not self._processedqueue.empty():
            processed_frames.append(self._processedqueue.get())
        return processed_frames
    
    def stop(self):
        self._stop_threads = True
        
session.close()