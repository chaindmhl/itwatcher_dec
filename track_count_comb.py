import queue, os, time, cv2, math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
physical_devices = tf.config.experimental.list_physical_devices('GPU')

if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tracking.deepsort_tric.deep_sort import preprocessing, nn_matching
from tracking.deepsort_tric.deep_sort.detection1 import Detection
from tracking.deepsort_tric.deep_sort.tracker import Tracker
from tracking.deepsort_tric.tools import generate_detections as gdet
import tracking.deepsort_tric.core.utils as utils
from tracking.deepsort_tric.core.config_tc import cfg
from tracking.models import VehicleLog
from collections import Counter, deque
import numpy as np
from datetime import date
from tracking.deepsort_tric.helper.detect_color import Detect_Color

stop_threads = False
detect_color = Detect_Color()

class Track_Count():
    def __init__(self, file_counter_log_name, framework='tf', weights='/home/itwatcher/Desktop/Itwatcher/restricted_yolov4_deepsort/checkpoints/yolov4-416',
                size=416, tiny=False, model='yolov4', video='./data/videos/cam0.mp4',
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
        self._allowed_classes = allowed_classes if allowed_classes else []
        self._detect_line_position = detection_line[0]
        self._detect_line_angle = detection_line[1]
        self._queue = frame_queue
        self._processedqueue = processed_queue
        self._time = processing_time
        self._stop_threads = False 

        self.total_counter = 0
        self.hwy_count = 0
        self.msu_count = 0
        self.sm_count = 0
        self.oval_count = 0
        self.class_counts = 0

    def get_total_counter(self):
        return self.total_counter
    
    def get_hwy_count(self):
        return self.hwy_count

    def get_msu_count(self):
        return self.msu_count
    
    def get_sm_count(self):
        return self.sm_count
    
    def get_oval_count(self):
        return self.oval_count
    
    def get_class_counts(self):
        return self.class_counts
    
    def _intersect(self, A, B, C, D):
        return self._ccw(A,C,D) != self._ccw(B, C, D) and self._ccw(A,B,C) != self._ccw(A,B,D)

    def _ccw(self, A,B,C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def _vector_angle(self, midpoint, previous_midpoint):
        x = midpoint[0] - previous_midpoint[0]
        y = midpoint[1] - previous_midpoint[1]
        return math.degrees(math.atan2(y, x))
    
    def _distance_point_to_line(self, point, line_start, line_end):
        # Calculate the distance from point to the line defined by line_start and line_end
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        numerator = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
        denominator = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distance = numerator / denominator
        return distance
    
    def _plate_within_roi(self, bbox, roi_vertices):
        # Calculate the center of the bounding box
        xmin, ymin, xmax, ymax = map(int, bbox)
        bbox_center = ((xmin + xmax) // 2, (ymin + ymax) // 2)
        
        # Check if the bbox_center is within the polygonal ROI
        roi_polygon = np.array(roi_vertices, dtype=np.int32)
        is_within_roi = cv2.pointPolygonTest(roi_polygon, bbox_center, False) >= 0
        
        return is_within_roi

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
        # north
        x1, y1 = 0, 600
        x2, y2 = 2400, 500
        lines.append([(x1, y1), (x2, y2)])
        
         # north
        x3, y3 = 0, 600
        x4, y4 = 2400, 500
        lines.append([(x3, y3), (x4, y4)])
        # west
        # xa, ya = 0, 750
        # xb, yb = 500, 1500
        xa, ya = 595, 1001
        xb, yb = 1559, 1437
        lines.append([(xa, ya), (xb, yb)])
        # east
        xc, yc = 2000, 0
        xd, yd = 2100, 1000
        lines.append([(xc, yc), (xd, yd)])

        # south
        xe, ye = 0, 1132
        xf, yf = 2009, 1035
        lines.append([(xe, ye), (xf, yf)])

        return lines
    
    def _define_and_draw_roi(self, frame):
   
        # Define the vertices of the ROI
        roi_vertices = [
            (0, 734),                   # Top-left
            (frame.shape[1], 0),        # Top-right
            (frame.shape[1], frame.shape[0]),  # Bottom-right
            (0, frame.shape[0])         # Bottom-left
        ]

        # Convert the vertices to a NumPy array
        roi_vertices_np = np.array(roi_vertices, dtype=np.int32).reshape((-1, 1, 2))

        # Draw the polygonal ROI on the frame
        cv2.polylines(frame, [roi_vertices_np], isClosed=True, color=(0, 255, 0), thickness=2)

        return roi_vertices_np

    def _process_tracked_objects(self, frame, tracker, memory, already_counted, already_passed, passed_counter, class_counts, lines, roi_vertices, detected_colors):
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            class_name = track.get_class()
            midpoint = track.tlbr_midpoint(bbox)
            self._update_memory(track.track_id, memory, midpoint)

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            # cv2.putText(frame, f"Vehicle Type: {class_name}", (int(bbox[0]), int(bbox[1]) - 10), 0, 0.7e-3 * frame.shape[0], (255, 125, 125), 2)
            text_to_display = str(class_name)
            if track.track_id in detected_colors:
                color_text = detected_colors[track.track_id]
                # text_to_display += f" - {color_text}"
                text_to_display += f" - {color_text}"

            # Display the text
            cv2.putText(frame, text_to_display, 
                        (int(bbox[0]), int(bbox[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1e-3 * frame.shape[0], 
                        (0, 0, 255), 
                        2)

            if self._plate_within_roi(bbox, roi_vertices) and track.track_id not in already_counted:
                class_counts[class_name] += 1
                self.total_counter += 1
                already_counted.append(track.track_id)

            self._check_lines_and_update_counters(
                frame, track, midpoint, memory[track.track_id][0], lines, passed_counter, already_passed, detected_colors
            )

    def _update_memory(self, track_id, memory, midpoint):
        if track_id not in memory:
            memory[track_id] = deque(maxlen=2)
        memory[track_id].append(midpoint)

    def _check_lines_and_update_counters(self, frame, track, midpoint, previous_midpoint, lines, passed_counter, already_passed, detected_colors):
        for line, direction in zip(lines, passed_counter.keys()):
            if self._intersect(midpoint, previous_midpoint, line[0], line[1]):
                self._process_detection(frame, track, detected_colors)

                angle = self._vector_angle(midpoint, previous_midpoint)
                if self._valid_direction(direction, angle, track.track_id, already_passed):
                    passed_counter[direction] += 1
                    already_passed.append(track.track_id)

    def _process_detection(self, frame, track, detected_colors):
        try:
            bbox = track.to_tlbr()
            xmin, ymin, xmax, ymax = map(int, bbox)
            vehicle_img = frame[ymin:ymax, xmin:xmax]
            # vehicle_img = cv2.cvtColor(vehicle_img, cv2.COLOR_RGB2BGR)
            vehicle_resized = cv2.resize(vehicle_img, (2000, 600), interpolation=cv2.INTER_LANCZOS4)

            prediction = detect_color.infer(vehicle_resized, track.track_id)
            if prediction["detected_class"]:
                detected_colors[track.track_id] = prediction["detected_class"]
        except cv2.error as e:
            print(f"Error during color detection: {e}")

    def _valid_direction(self, direction, angle, track_id, already_passed):
        conditions = {
            "Going North": angle > 0 and track_id not in already_passed,
            "Going West": angle < 0 and track_id not in already_passed,
            "Going East": angle < 0 and track_id not in already_passed,
            "Going South": angle < 0 and track_id not in already_passed,
        }
        return conditions.get(direction, False)

    def _process_frame(self, frame, input_size, infer, encoder, tracker, memory, already_counted, already_passed, passed_counter, class_counts, nms_max_overlap=0.1, detected_colors={}):
        batch_data, original_h, original_w = self._preprocess_frame(frame, input_size)
        pred_bbox = infer(batch_data)

        bboxes, scores, classes, num_objects = self._extract_detections(pred_bbox, original_h, original_w)
        detections = self._filter_and_encode_detections(frame, bboxes, scores, classes, encoder)
        detections = self._apply_nms(detections, nms_max_overlap)

        self._track_objects(tracker, detections)

        lines = self._define_lines(frame)
        roi_vertices = self._define_and_draw_roi(frame)

        self._process_tracked_objects(frame, tracker, memory, already_counted, already_passed, passed_counter, class_counts, lines, roi_vertices, detected_colors)

        # Log and display counts (this logic should be in separate functions if needed)
        self._log_vehicle_count(passed_counter, class_counts)
        self._display_class_counts(frame, class_counts)
        self._display_total_vehicle_count(frame)
        self._cleanup_memory(memory)

        return np.asarray(frame)

    def _log_vehicle_count(self, passed_counter, class_counts):
        vehicle_logs = []
        for direction, cnt in passed_counter.items():
            if direction == "Going North":
                hwy_count = cnt
            elif direction == "Going West":
                msu_count = cnt
            elif direction == "Going East":
                sm_count = cnt
            elif direction == "Going South":
                oval_count = cnt

        current_date = date.today()
        existing_log = VehicleLog.objects.filter(date=current_date).first()

        if existing_log:
            existing_log.total_count = self.total_counter
            existing_log.hwy_count = hwy_count
            existing_log.msu_count = msu_count
            existing_log.sm_count = sm_count
            existing_log.oval_count = oval_count
            existing_log.class_counts = class_counts
            existing_log.save()
        else:
            vehicle_log = VehicleLog.objects.create(
                date=current_date,
                filename=self._file_counter_log_name,
                total_count=self.total_counter,
                hwy_count=hwy_count,
                msu_count=msu_count,
                sm_count=sm_count,
                oval_count=oval_count,
                class_counts=class_counts,
            )
            vehicle_log.save()

    def _cleanup_memory(self, memory):
        for track_id in list(memory.keys()):
            if len(memory[track_id]) > 50:
                del memory[track_id]
    

    def _display_class_counts(self, frame, class_counts):
       
        y = 0.6 * frame.shape[0]  # Y-coordinate for starting text position
        for cls in class_counts:
            class_count = class_counts[cls]
            cv2.putText(frame, f"{cls}: {class_count}", 
                        (int(0.02 * frame.shape[1]), int(y)), 
                        2, 1.0e-3 * frame.shape[0], 
                        (255, 0, 0), 3)
            y += 0.04 * frame.shape[0]  # Adjust the Y-coordinate for the next class

    def _display_total_vehicle_count(self, frame):
        """
        Display the total vehicle count on the frame.
        """
        cv2.putText(frame, f"Total Vehicle Count: {self.total_counter}", 
                    (int(0.75 * frame.shape[1]), int(0.03 * frame.shape[0])), 
                    2, 1.0e-3 * frame.shape[0], 
                    (255, 0, 0), 3)

        # Iterate through passed_counter to display counts for each direction
        # h = 0.8 * frame.shape[0]
        # for direction, cnt in passed_counter.items():
        #     class_count_str = f"{direction}:{cnt}"
            
        #     # Display the count for the direction
        #     cv2.putText(frame, class_count_str, (int(0.02 * frame.shape[1]), int(h)), 0, 1.3e-3 * frame.shape[0], (255, 0, 0), 3)
        #     h += 0.05 * frame.shape[0]

    # Prepare the text to display: "class_name - detected_color"
            

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
        class_counts = Counter()
        already_counted = deque(maxlen=50) 
        already_passed = deque(maxlen=50)
        passed_counter = {
                        "Going North": 0,
                        "Going West": 0,
                        "Going East": 0,
                        "Going South": 0
                    }
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

                result = self._process_frame(frame, input_size, infer, encoder, tracker, memory, already_counted, already_passed, passed_counter, class_counts)

                end_time = time.time()
                processing_time = end_time - start_time
                total_processing_time += processing_time
                num_frames_processed += 1

                if result is not None and len(result) > 0:
                    self._processedqueue.put(result)

                if result is not None and result.size > 0:
                    out.write(result)
                else:
                    print("Warning: Attempted to write an invalid frame.")

        finally:
            # Ensure the VideoWriter is released at the end
            if out is not None:
                out.release()
            vid.release()

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