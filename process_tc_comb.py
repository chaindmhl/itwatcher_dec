from django.conf import settings
import datetime, requests, os, cv2, queue, threading
from django.core.files import File
from tracking.deepsort_tric.track_count_comb import Track_Count
from tracking.deepsort_tric.core.config_tc import cfg
import tracking.deepsort_tric.core.utils as utils


REQUEST_URL = f"http://{settings.HOST}:8000/"


def process_trackcount_comb(video_path=None, livestream_url=None, is_live_stream=False, video_stream=None):

    processed_frames = []
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)
    allowed_classes = list(class_names.values())
    # Tricycle Detection and Tracking
    if video_path or directory_path:
        video_files = []
        
        # Check if `video_path` is a directory
        if os.path.isdir(video_path):
            # Gather all video files in the directory
            video_files = [
                os.path.join(video_path, f)
                for f in os.listdir(video_path)
                if f.lower().endswith(('.mp4', '.avi', '.mov'))  # Add other formats if needed
            ]
            if not video_files:
                raise ValueError(f"No video files found in directory: {video_path}")
        else:
            # Treat `video_path` as a single file
            video_files = [video_path]
    
        # Iterate through all video files (either single or from directory)
        for video_file in video_files:
            print(f"Processing video: {video_file}")

        # Create a folder to store the output frames
        output_folder_path = os.path.join(settings.MEDIA_ROOT, 'tracked_count')
        os.makedirs(output_folder_path, exist_ok=True)

        # Specify the filename and format of the output video
        output_video_path = os.path.join(output_folder_path, f"tracked_{os.path.basename(video_path)}")

        # Create an instance of the VehiclesCounting class
        tc = Track_Count(file_counter_log_name='itwatcher_count.log',
                              framework='tf',
                              weights='/home/icebox/itwatcher_api/tracking/deepsort_tric/checkpoints/itwatcher',
                              size=416,
                              tiny=False,
                              model='yolov4',
                              video=video_file,
                              output=output_video_path,
                              output_format='XVID',
                              iou=0.45,
                              score=0.5,
                              dont_show=False,
                              info=False,
                              detection_line=(0.5, 0),
                              frame_queue = queue.Queue(maxsize=1200),
                              processed_queue=queue.Queue(maxsize=100),
                              allowed_classes=allowed_classes)

        # Start producer and consumer threads
        producer_thread = threading.Thread(target=tc.producer)
        consumer_thread = threading.Thread(target=tc.consumer)


        producer_thread.start()
        consumer_thread.start()

        # Retrieve frames from the processed_queue in real-time
        while producer_thread.is_alive() or consumer_thread.is_alive():
            try:
                processed_frame = tc._processedqueue.get(timeout=1)  # Wait for a frame for 1 second
                yield processed_frame  # Yield the processed frame
            except queue.Empty:
                continue

        # Ensure the threads are terminated
        tc.stop()
        
        # Wait for both threads to finish
        producer_thread.join()
        consumer_thread.join()
        print("stop threads")

        # Retrieve any remaining frames in the queue
        while not tc._processedqueue.empty():
            processed_frame = tc._processedqueue.get()
            yield processed_frame
        
    return processed_frames