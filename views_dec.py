from django.contrib.auth.models import User
from rest_framework.permissions import IsAuthenticated
from django.views import View
from PIL import Image
from io import BytesIO
from django.core.files import File
from django.utils.timezone import now


from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from django.shortcuts import render,redirect, get_object_or_404
from django.views.decorators.http import require_POST
from rest_framework import viewsets, generics, status
from rest_framework.response import Response
from tracking.serializers import (
    VideoSerializers,
    ProcessVideoSerializers,
    LPRSerializers,
    CountLogSerializer,
    VehicleLogSerializer,
    DownloadRequestSerializer,
    VideoSerializers
)
from tracking.models import Video, PlateLog, CountLog, ColorLog, VehicleLog, DownloadRequest, ViolationLog
from tracking.process_tc_trike import process_trackcount_trike
from tracking.process_tc_all import process_trackcount_all
from tracking.process_tc_comb import process_trackcount_comb
from tracking.process_lpr_trike import process_lpr_trike
from tracking.process_lpr_all import process_alllpr
from tracking.process_lpr_comb import process_lpd_comb
from tracking.process_redlight import process_redlight
from tracking.process_blocking import process_blocking
from tracking.process_speeding import process_overspeeding
import ffmpeg, glob
from django.utils.text import get_valid_filename

from django.http import StreamingHttpResponse, Http404, HttpResponse
from django.utils.timezone import make_naive
from django.conf import settings
from tracking.deepsort_tric.helper.light_state import set_current_light_state, get_current_light_state
# from .config import show_disp
# import configparser

# config = configparser.ConfigParser()
# # Read the configuration file
# config.read('itwatcher.cfg')


# Define your Django Rest Framework view
from rest_framework.response import Response
import matplotlib, shutil
matplotlib.use('Agg')  # Use the non-GUI backend 'Agg'
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
import io, os, base64, subprocess, socket, cv2, json, logging
from django.db.models import Count, F
from datetime import timedelta, datetime
from tracking.forms import SignUpForm
from django.contrib.auth.views import LoginView as AuthLoginView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import render
import time
from rest_framework.decorators import action


# Create your views here.
class DarknetTrainView(View):
    def post(self, request, *args, **kwargs):
        # Get parameters from the form
        data_path = request.POST.get('data_path')
        cfg_path = request.POST.get('cfg_path')
        weight_path = request.POST.get('weight_path')
        # ... add other parameters as needed

        # Set the working directory to the Darknet folder
        darknet_path = '/home/icebox/darknet'  # Replace with the actual path to your Darknet folder
        os.chdir(darknet_path)

        # Run Darknet training command
        command = f'./darknet detector train {data_path} {cfg_path} {weight_path}'

        try:
            result = subprocess.check_output(command, shell=True)
            return HttpResponse(result)
        except subprocess.CalledProcessError as e:
            return HttpResponse(str(e))

    def get(self, request, *args, **kwargs):
        return render(request, 'html_files/train.html')

class CustomLoginView(AuthLoginView):
    template_name = 'html_files/login.html'

class SignupView(View):

    def get(self, request):
        form = SignUpForm()
        return render(request, 'html_files/signup.html', {'form': form})

    def post(self, request):
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')  # Redirect to the login page after successful signup
        return render(request, 'html_files/signup.html', {'form': form})

class LPRView(View):
    
    def get(self, request):
        users = User.objects.all()
        videos = Video.objects.all()
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        context = {'users': users, 'videos': videos, 'hostname': hostname, 'ip_address': ip_address}  # Optionally pass hostname and ip_address to the template
        return render(request, 'html_files/lpr.html', context)



class TrackCountView(View):
    
    def get(self, request):
        users = User.objects.all()
        videos = Video.objects.all()
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        context = {'users': users, 'videos': videos, 'hostname': hostname, 'ip_address': ip_address}
        return render(request, 'html_files/track_count.html', context)  

class ColorRecognitionView(View):
    
    def get(self, request):
        users = User.objects.all()
        videos = Video.objects.all()
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        context = {'users': users, 'videos': videos, 'hostname': hostname, 'ip_address': ip_address}
        return render(request, 'html_files/color.html', context)

class VioDetectionView(View):
    
    def get(self, request):
        users = User.objects.all()
        videos = Video.objects.all()
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        context = {'users': users, 'videos': videos, 'hostname': hostname, 'ip_address': ip_address}
        return render(request, 'html_files/violation.html', context) 

class MyView(LoginRequiredMixin, View):
    login_url = '/login/'  # Set the URL where unauthenticated users are redirected

    def get(self, request):
        users = User.objects.all()
        videos = Video.objects.all()
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        context = {'users': users, 'videos': videos, 'hostname': hostname, 'ip_address': ip_address}
        return render(request, 'html_files/index.html', context)

class UploadView(View):
    
    permission_classes = [IsAuthenticated]

    def get(self, request):
        users = User.objects.all()
        videos = Video.objects.all()
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        context = {'users': users, 'videos': videos, 'hostname': hostname, 'ip_address': ip_address}
        return render(request, 'html_files/upload_video.html', context)

class CountLogViewSet(viewsets.ModelViewSet):
    queryset = CountLog.objects.all()
    serializer_class = CountLogSerializer

class VehicleLogViewSet(viewsets.ModelViewSet):
    queryset = VehicleLog.objects.all()
    serializer_class = VehicleLogSerializer
       
# Create your views here.
class VideoUploadViewSet(viewsets.ModelViewSet):
    """
    Uploads a File
    """

    queryset = Video.objects.all()
    serializer_class = VideoSerializers
    permission_classes = [IsAuthenticated]

class Streaming:
    def stream_processed_frames(self, processed_frames, request):

        # Get show_display from POST data (or default to False if not provided)
        show_disp = request.POST.get('show_display', 'false').lower() == 'true'

        if not processed_frames:
            return StreamingHttpResponse(status=204)

        def generate():
            for frame in processed_frames:
                if not show_disp:
                    continue  # Skip the frame if show_disp is False
                try:
                    resized_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC)
                    success, buffer = cv2.imencode('.jpg', resized_frame)
                    
                    while not success:
                        print("Failed to encode frame, retrying...")
                        success, buffer = cv2.imencode('.jpg', resized_frame)
                        time.sleep(0.01)
                    
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    yield f"data: {frame_base64}\n\n"
                    
                    time.sleep(0.05)  # Control the frame rate (20 FPS)
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue

        response = StreamingHttpResponse(generate(), content_type='text/event-stream')
        response['Cache-Control'] = 'no-cache'
        response['Access-Control-Allow-Origin'] = '*'  # Update with specific domain if needed
        return response

class ProcessTrikeViewSet(viewsets.ViewSet, Streaming):
    """
    Perform tricycle Detection in Videos
    """

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("livestream_url")
            print("URL:", video_path)

            # If a specific video is provided in the request
            if video_path:
                processed_frames = process_trackcount_trike(video_path=video_path)  # Process the given video
                # Update the database to mark the video as processed
                Video.objects.filter(file=video_path).update(
                    processed_tc_trike=True, last_processed_tc_trike=now()
                )
                return self.stream_processed_frames(processed_frames, request)

            # If no specific video path is provided, stream all videos from the database
            else:
                # videos = Video.objects.all()[:5]  # Limit to 5 videos for testing
                # Fetch videos that are not yet processed
                videos = Video.objects.filter(processed_tc_trike=False)

                # Generate frames one-by-one, processing and streaming immediately
                def generate():
                    for video in videos:
                        video_path = video.file.path  # Get the path of each video file in the database
                        print(f"Processing video: {video.file.path}")
                        
                        # Process the video and stream frames as they are processed
                        processed_frames = process_trackcount_trike(video_path=video_path)
                        if not processed_frames:
                            print(f"No frames processed for video: {video.file.path}")
                        
                        # Mark the video as processed after successful processing
                        video.processed_tc_trike = True
                        video.last_processed_tc_trike = now()
                        video.save()

                        for frame in processed_frames:
                            yield frame  # Yield each frame immediately to be streamed
                            # time.sleep(0.05)  # Optional: Add a small delay to control the streaming rate

                return self.stream_processed_frames(generate(), request)
                # return StreamingHttpResponse(generate(), content_type='text/event-stream')

        else:
            print("Invalid serializer data:", serializer.errors)
            return JsonResponse(serializer.errors, status=status.HTTP_400_BAD_REQUEST)




class CatchAllViewSet(viewsets.ViewSet, Streaming):
    """
    Perform Vehicle Detection in Videos
    """

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("livestream_url")
            print("URL:", video_path)

            # If a specific video is provided in the request
            if video_path:
                processed_frames = process_trackcount_all(video_path=video_path)  # Process the given video
                # Update the database to mark the video as processed
                Video.objects.filter(file=video_path).update(
                    processed_tc_catchall=True, last_processed_tc_catchall=now()
                )
                return self.stream_processed_frames(processed_frames, request)

            # If no specific video path is provided, stream all videos from the database
            else:
                videos = Video.objects.filter(processed_tc_catchall=False)

                # Generate frames one-by-one, processing and streaming immediately
                def generate():
                    for video in videos:
                        video_path = video.file.path  # Get the path of each video file in the database
                        print(f"Processing video: {video.file.path}")
                        
                        # Process the video and stream frames as they are processed
                        processed_frames = process_trackcount_all(video_path=video_path)
                        if not processed_frames:
                            print(f"No frames processed for video: {video.file.path}")
                        
                         # Mark the video as processed after successful processing
                        video.processed_tc_catchall = True
                        video.last_processed_tc_catchall = now()
                        video.save()

                        for frame in processed_frames:
                            yield frame  # Yield each frame immediately to be streamed

                return self.stream_processed_frames(generate(), request)

        else:
            print("Invalid serializer data:", serializer.errors)
            return JsonResponse(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



class CombiViewSet(viewsets.ViewSet, Streaming):
    """
    Perform Vehicle Detection (Trike and Vehicle) in Videos
    """

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("livestream_url")
            print("URL:", video_path)

            # If a specific video is provided in the request
            if video_path:
                processed_frames = process_trackcount_comb(video_path=video_path)  # Process the given video
                # Update the database to mark the video as processed
                Video.objects.filter(file=video_path).update(
                    processed_tc_itwatcher=True, last_processed_tc_itwatcher=now()
                )
                return self.stream_processed_frames(processed_frames, request)

            # If no specific video path is provided, stream all videos from the database
            else:
                videos = Video.objects.filter(processed_tc_itwatcher=False)  

                # Generate frames one-by-one, processing and streaming immediately
                def generate():
                    for video in videos:
                        video_path = video.file.path  # Get the path of each video file in the database
                        print(f"Processing video: {video.file.path}")
                        
                        # Process the video and stream frames as they are processed
                        processed_frames = process_trackcount_comb(video_path=video_path)
                        if not processed_frames:
                            print(f"No frames processed for video: {video.file.path}")
                        
                        # Mark the video as processed after successful processing
                        video.processed_tc_itwatcher = True
                        video.last_processed_tc_itwatcher = now()
                        video.save()

                        for frame in processed_frames:
                            yield frame  # Yield each frame immediately to be streamed

                return self.stream_processed_frames(generate(), request)
                
        else:
            print("Invalid serializer data:", serializer.errors)
            return JsonResponse(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


        

class LPRTrikeViewSet(viewsets.ViewSet, Streaming):
    """
    Perform LPR-trike in Videos
    """

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("livestream_url")
            print("URL:", video_path)

            # If a specific video is provided in the request
            if video_path:
                processed_frames = process_lpr_trike(video_path=video_path)  # Process the given video
                # Update the database to mark the video as processed
                Video.objects.filter(file=video_path).update(
                    processed_lpr_trike=True, last_processed_lpr_trike=now()
                )
                return self.stream_processed_frames(processed_frames, request)

            # If no specific video path is provided, stream all videos from the database
            else:
                videos = Video.objects.filter(processed_lpr_trike=False) # Limit to 5 videos for testing

                # Generate frames one-by-one, processing and streaming immediately
                def generate():
                    for video in videos:
                        video_path = video.file.path  # Get the path of each video file in the database
                        print(f"Processing video: {video.file.path}")
                        
                        # Process the video and stream frames as they are processed
                        processed_frames = process_lpr_trike(video_path=video_path)
                        if not processed_frames:
                            print(f"No frames processed for video: {video.file.path}")
                        
                        # Mark the video as processed after successful processing
                        video.processed_lpr_trike = True
                        video.last_processed_lpr_trike = now()
                        video.save()

                        for frame in processed_frames:
                            yield frame  # Yield each frame immediately to be streamed

                return self.stream_processed_frames(generate(), request)

        else:
            print("Invalid serializer data:", serializer.errors)
            return JsonResponse(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LPRAllViewSet(viewsets.ViewSet, Streaming):
    """
    Perform LPR-all in Videos
    """

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("livestream_url")
            print("URL:", video_path)

            # If a specific video is provided in the request
            if video_path:
                processed_frames = process_alllpr(video_path=video_path)  # Process the given video
                # Update the database to mark the video as processed
                Video.objects.filter(file=video_path).update(
                    processed_lpr_catchall=True, last_processed_lpr_catchall=now()
                )
                return self.stream_processed_frames(processed_frames, request)

            # If no specific video path is provided, stream all videos from the database
            else:
                videos = Video.objects.filter(processed_lpr_catchall=False)

                # Generate frames one-by-one, processing and streaming immediately
                def generate():
                    for video in videos:
                        video_path = video.file.path  # Get the path of each video file in the database
                        print(f"Processing video: {video.file.path}")
                        
                        # Process the video and stream frames as they are processed
                        processed_frames = process_alllpr(video_path=video_path)
                        if not processed_frames:
                            print(f"No frames processed for video: {video.file.path}")
                        
                        # Mark the video as processed after successful processing
                        video.processed_lpr_catchall = True
                        video.last_processed_lpr_catchall = now()
                        video.save()

                        for frame in processed_frames:
                            yield frame  # Yield each frame immediately to be streamed
                            # time.sleep(0.05)  # Optional: Add a small delay to control the streaming rate

                return self.stream_processed_frames(generate(), request)
                # return StreamingHttpResponse(generate(), content_type='text/event-stream')

        else:
            print("Invalid serializer data:", serializer.errors)
            return JsonResponse(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    
class LPRCombiViewSet(viewsets.ViewSet, Streaming):
    """
    Perform LPR_comb in Videos
    """

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("livestream_url")
            print("URL:", video_path)

            # If a specific video is provided in the request
            if video_path:
                processed_frames = process_lpd_comb(video_path=video_path)  # Process the given video
                # Update the database to mark the video as processed
                Video.objects.filter(file=video_path).update(
                    processed_lpr_itwatcher=True, last_processed_lpr_itwatcher=now()
                )
                return self.stream_processed_frames(processed_frames, request)

            # If no specific video path is provided, stream all videos from the database
            else:
                videos = Video.objects.filter(processed_lpr_itwatcher=False)

                # Generate frames one-by-one, processing and streaming immediately
                def generate():
                    for video in videos:
                        video_path = video.file.path  # Get the path of each video file in the database
                        print(f"Processing video: {video.file.path}")
                        
                        # Process the video and stream frames as they are processed
                        processed_frames = process_lpd_comb(video_path=video_path)
                        if not processed_frames:
                            print(f"No frames processed for video: {video.file.path}")
                        
                        # Mark the video as processed after successful processing
                        video.processed_lpr_itwatcher = True
                        video.last_processed_lpr_itwatcher = now()
                        video.save()

                        for frame in processed_frames:
                            yield frame  # Yield each frame immediately to be streamed
                            # time.sleep(0.05)  # Optional: Add a small delay to control the streaming rate

                return self.stream_processed_frames(generate(), request)
                # return StreamingHttpResponse(generate(), content_type='text/event-stream')

        else:
            print("Invalid serializer data:", serializer.errors)
            return JsonResponse(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class RedLightViewSet(viewsets.ViewSet, Streaming):
    """
    Perform Swerving Detection in Videos
    """

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("livestream_url")
            print("URL:", video_path)

            # If a specific video is provided in the request
            if video_path:
                processed_frames = process_redlight(video_path=video_path)  # Process the given video
                # Update the database to mark the video as processed
                Video.objects.filter(file=video_path).update(
                    processed_brl=True, last_processed_brl=now()
                )
                return self.stream_processed_frames(processed_frames, request)

            # If no specific video path is provided, stream all videos from the database
            else:
                videos = Video.objects.filter(processed_brl=False) 

                # Generate frames one-by-one, processing and streaming immediately
                def generate():
                    for video in videos:
                        video_path = video.file.path  # Get the path of each video file in the database
                        print(f"Processing video: {video.file.path}")
                        
                        # Process the video and stream frames as they are processed
                        processed_frames = process_redlight(video_path=video_path)
                        if not processed_frames:
                            print(f"No frames processed for video: {video.file.path}")
                        
                        # Mark the video as processed after successful processing
                        video.processed_brl = True
                        video.last_processed_brl = now()
                        video.save()

                        for frame in processed_frames:
                            yield frame  # Yield each frame immediately to be streamed
                            # time.sleep(0.05)  # Optional: Add a small delay to control the streaming rate

                return self.stream_processed_frames(generate(), request)
                # return StreamingHttpResponse(generate(), content_type='text/event-stream')

        else:
            print("Invalid serializer data:", serializer.errors)
            return JsonResponse(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['post'])
    def control_traffic_light(self, request):
        color = request.data.get('color')
        print(f"Received request to change light color to: {color}")
        if color in ['red', 'yellow', 'green']:
            set_current_light_state(color)
            print(f"Light color successfully changed to: {get_current_light_state()}")
            return Response({'status': 'success', 'current_light_state': get_current_light_state()}, status=status.HTTP_200_OK)
        else:
            print("Invalid color received")
            return Response({'error': 'Invalid color'}, status=status.HTTP_400_BAD_REQUEST)
        

class BlockingViewSet(viewsets.ViewSet, Streaming):
    """
    Perform Blocking Detection in Videos
    """

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("livestream_url")
            print("URL:", video_path)

            # If a specific video is provided in the request
            if video_path:
                processed_frames = process_blocking(video_path=video_path)  # Process the given video
                # Update the database to mark the video as processed
                Video.objects.filter(file=video_path).update(
                    processed_bpl=True, last_processed_bpl=now()
                )
                return self.stream_processed_frames(processed_frames, request)

            # If no specific video path is provided, stream all videos from the database
            else:
                videos = Video.objects.filter(processed_bpl=False)

                # Generate frames one-by-one, processing and streaming immediately
                def generate():
                    for video in videos:
                        video_path = video.file.path  # Get the path of each video file in the database
                        print(f"Processing video: {video.file.path}")
                        
                        # Process the video and stream frames as they are processed
                        processed_frames = process_blocking(video_path=video_path)
                        if not processed_frames:
                            print(f"No frames processed for video: {video.file.path}")
                        
                        # Mark the video as processed after successful processing
                        video.processed_bpl = True
                        video.last_processed_bpl = now()
                        video.save()

                        for frame in processed_frames:
                            yield frame  # Yield each frame immediately to be streamed
                            # time.sleep(0.05)  # Optional: Add a small delay to control the streaming rate

                return self.stream_processed_frames(generate(), request)
                # return StreamingHttpResponse(generate(), content_type='text/event-stream')

        else:
            print("Invalid serializer data:", serializer.errors)
            return JsonResponse(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class SpeedViewSet(viewsets.ViewSet, Streaming):
    """
    Perform Overspeeding Detection in Videos
    """

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("livestream_url")
            print("URL:", video_path)

            # If a specific video is provided in the request
            if video_path:
                processed_frames = process_overspeeding(video_path=video_path)  # Process the given video
                # Update the database to mark the video as processed
                Video.objects.filter(file=video_path).update(
                    processed_os=True, last_processed_os=now()
                )
                return self.stream_processed_frames(processed_frames, request)

            # If no specific video path is provided, stream all videos from the database
            else:
                videos = Video.objects.filter(processed_os=False) 

                # Generate frames one-by-one, processing and streaming immediately
                def generate():
                    for video in videos:
                        video_path = video.file.path  # Get the path of each video file in the database
                        print(f"Processing video: {video.file.path}")
                        
                        # Process the video and stream frames as they are processed
                        processed_frames = process_overspeeding(video_path=video_path)
                        if not processed_frames:
                            print(f"No frames processed for video: {video.file.path}")
                        
                        # Mark the video as processed after successful processing
                        video.processed_os = True
                        video.last_processed_os = now()
                        video.save()

                        for frame in processed_frames:
                            yield frame  # Yield each frame immediately to be streamed
                            # time.sleep(0.05)  # Optional: Add a small delay to control the streaming rate

                return self.stream_processed_frames(generate(), request)
                # return StreamingHttpResponse(generate(), content_type='text/event-stream')

        else:
            print("Invalid serializer data:", serializer.errors)
            return JsonResponse(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ColorView(View):

    def get(self, request):
        color_logs = ColorLog.objects.all()
        context = {
            'color_logs': color_logs,
        }
        return render(request, '/home/icebox/itwatcher_api/tracking/html_files/display_color.html', context)
    def post(self, request):
        # Handle POST requests if needed
        pass


class PlateView(View):
    def get(self, request):
        # Select distinct filenames with timestamps within a 5-minute range and having three or more characters
        distinct_filenames = PlateLog.objects.filter(
            timestamp__range=(F('timestamp') - timedelta(minutes=5), F('timestamp') + timedelta(minutes=5)),
            filename__in=PlateLog.objects.values('filename').annotate(count=Count('filename')).filter(count__gte=3).values('filename')
        ).values('filename').distinct()

        # Create a list of IDs to exclude
        exclude_ids = PlateLog.objects.exclude(
            filename__in=distinct_filenames
        ).values('id')

        # Exclude the unwanted entries
        PlateLog.objects.exclude(id__in=exclude_ids).delete()

        # Retrieve the remaining PlateLog records
        plate_logs = PlateLog.objects.all()

        # Get the hostname and IP address
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        
        context = {
            'plate_logs': plate_logs,
            'hostname': hostname,
            'ip_address': ip_address,
        }
        return render(request, '/home/icebox/itwatcher_api/tracking/html_files/display_plates.html', context)

    def post(self, request):
        # Handle POST requests if needed
        pass

class FrameColorView(View):

    def view_colorframe(request, log_id):
        # Retrieve the PlateLog instance based on the log_id
        color_log = ColorLog.objects.get(id=log_id)
        context = {
            'color_log': color_log,
        }
        return render(request, '/home/icebox/itwatcher_api/tracking/html_files/view_colorframe.html', context)
    
class FrameViolationView(View):

    def view_violationframe(request, log_id):
        # Retrieve the PlateLog instance based on the log_id
        vio_log = ViolationLog.objects.get(id=log_id)

        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        context = {
            'vio_log': vio_log,
            'hostname': hostname,
            'ip_address': ip_address,
        }
        return render(request, '/home/icebox/itwatcher_api/tracking/html_files/view_violationframe.html', context)
    
class ViolationView(View):
    def get(self, request):
        vio_logs = ViolationLog.objects.all()

        for log in vio_logs:
            log.timestamp_date = log.timestamp_date.strftime('%Y-%m-%d')
            log.timestamp_time = log.timestamp_time.strftime('%H:%M:%S')

        # Get the hostname and IP address
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)

        context = {
            'vio_logs': vio_logs,
            'hostname': hostname,
            'ip_address': ip_address,
        }
        return render(request, '/home/icebox/itwatcher_api/tracking/html_files/violation_list.html', context)
    def post(self, request):
        # Handle POST requests if needed
        pass


class FrameView(View):

    def view_frame(request, log_id):
        # Retrieve the PlateLog instance based on the log_id
        plate_log = PlateLog.objects.get(id=log_id)
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        

        context = {
            'plate_log': plate_log,
            'hostname': hostname,
            'ip_address': ip_address,

        }
        return render(request, '/home/icebox/itwatcher_api/tracking/html_files/view_frame.html', context)
    
class MapView(View):

    def view_camera_map(request):
        return render(request, '/home/icebox/itwatcher_api/tracking/html_files/view_camera_map.html')

@require_POST
def update_plate_number(request):
    data = json.loads(request.body)
    log_id = data.get('log_id')
    plate_number = data.get('plate_number')

    if log_id and plate_number:
        plate_log = get_object_or_404(PlateLog, id=log_id)
        plate_log.plate_number = plate_number
        plate_log.save()
        return HttpResponse(status=200)  # Successful update
    else:
        return HttpResponse(status=400)  # Bad request

class CountLogListView(View):

    template_name = 'html_files/count_log_list.html'

    def get(self, request, *args, **kwargs):
        count_logs = CountLog.objects.all()
        context = {'count_logs': count_logs}
        return render(request, self.template_name, context)
            
class VehicleLogListView(View):

    template_name = 'html_files/vehicle_log_list.html'

    def get(self, request, *args, **kwargs):
        vehicle_logs = VehicleLog.objects.all()
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        context = {'vehicle_logs': vehicle_logs,
                   'hostname': hostname,
                    'ip_address': ip_address,}
        return render(request, self.template_name, context)
    
class TrikeVehicleLogListView(View):

    template_name = 'html_files/trikeall_log_list.html'

    def get(self, request, *args, **kwargs):
        trikeall_logs = VehicleLog.objects.all()
        context = {'trikeall_logs': trikeall_logs}
        return render(request, self.template_name, context)

class TricycleCountGraphView(View):
    template_name = 'html_files/tricycle_count_graph.html'  # Path to your template

    def get(self, request, log_id):
        # Retrieve the log entry based on log_id
        log = CountLog.objects.get(id=log_id)  # Adjust this based on your model

        # Extract class names and counts from the log
        class_names = list(log.class_counts.keys())
        class_counts = list(log.class_counts.values())

        # Generate the bar graph
        bar_figure = plt.figure(figsize=(6, 4))
        plt.bar(class_names, class_counts)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Bar Graph')
        bar_buffer = io.BytesIO()
        plt.savefig(bar_buffer, format='png')
        plt.close(bar_figure)

        # Generate the pie chart
        pie_figure = plt.figure(figsize=(6, 4))
        plt.pie(class_counts, labels=class_names, autopct='%1.1f%%')
        plt.title('Pie Chart')
        pie_buffer = io.BytesIO()
        plt.savefig(pie_buffer, format='png')
        plt.close(pie_figure)

        # Generate the line graph
        line_figure = plt.figure(figsize=(6, 4))
        plt.plot(class_names, class_counts, marker='o')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Line Graph')
        line_buffer = io.BytesIO()
        plt.savefig(line_buffer, format='png')
        plt.close(line_figure)

        # Convert the buffer data to base64 for embedding in the HTML
        bar_graph_data = base64.b64encode(bar_buffer.getvalue()).decode()
        pie_graph_data = base64.b64encode(pie_buffer.getvalue()).decode()
        line_graph_data = base64.b64encode(line_buffer.getvalue()).decode()

        context = {
            'bar_graph_data': bar_graph_data,
            'pie_graph_data': pie_graph_data,
            'line_graph_data': line_graph_data,
            'log_id': log_id,
        }

        return render(request, self.template_name, context)

class VehicleCountGraphView(View):
    template_name = 'html_files/vehicle_count_graph.html'  # Path to your template

    def get(self, request, log_date, log_id):
        # Retrieve the log entry based on log_id
        log = VehicleLog.objects.get(id=log_id)  # Adjust this based on your model
        # logs = VehicleLog.objects.filter(timestamp__date=log_date)

        # Extract class names and counts from the log
        class_names = list(log.class_counts.keys())
        class_counts = list(log.class_counts.values())

        # Generate the bar graph
        bar_figure = plt.figure(figsize=(8, 6))
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'yellow', 'black', 'cyan','white','green', 'blue', 'violet']  # Add more colors if needed
        
        # Calculate cumulative counts for each vehicle type
        cumulative_counts = [0] * len(class_names)
        for i, count in enumerate(class_counts):
            plt.bar(log_date, count, bottom=cumulative_counts, color=colors[i], label=class_names[i])
            cumulative_counts[i] += count

        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.title('Bar Graph for {log_date}')
        bar_buffer = io.BytesIO()
        plt.savefig(bar_buffer, format='png')
        plt.close(bar_figure)
        bar_graph_data = base64.b64encode(bar_buffer.getvalue()).decode()

        # Generate the pie chart
        pie_figure = plt.figure(figsize=(6, 4))
        plt.pie(class_counts, labels=class_names, autopct='%1.1f%%')
        plt.title('Pie Chart')
        pie_buffer = io.BytesIO()
        plt.savefig(pie_buffer, format='png')
        plt.close(pie_figure)
        pie_graph_data = base64.b64encode(pie_buffer.getvalue()).decode()

        # Generate the line graph
        line_figure = plt.figure(figsize=(6, 4))
        plt.plot(class_names, class_counts, marker='o')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Line Graph')
        line_buffer = io.BytesIO()
        plt.savefig(line_buffer, format='png')
        plt.close(line_figure)
        line_graph_data = base64.b64encode(line_buffer.getvalue()).decode()

        context = {
            'bar_graph_data': bar_graph_data,
            'pie_graph_data': pie_graph_data,
            'line_graph_data': line_graph_data,
            'log_date': log_date,
            
        }

        return render(request, self.template_name, context)


class DownloadRequestListCreateView(generics.ListCreateAPIView):
    template_name = 'html_files/download_video.html'
    queryset = DownloadRequest.objects.all()
    serializer_class = DownloadRequestSerializer

    def get(self, request, *args, **kwargs):
        return render(request, self.template_name)

    def post(self, request, *args, **kwargs):
        start = '08:00:00'
        end = '07:59:59'
        script_path = '/home/icebox/itwatcher_api/tracking/hikvision/media_download.py'
        camera_ip = request.POST.get('camera_ip')
        start_date = request.POST.get('start_date')
        start_time = request.POST.get('start_time', start)
        end_date = request.POST.get('end_date')
        end_time = request.POST.get('end_time', end)
        content_type = request.POST.get('content_type')
        channel = request.POST.get('channel')

        try:
            # Ensure dates are properly parsed
            if start_date:
                start_datetime = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M:%S")
            else:
                start_datetime = datetime.now()

            if not end_date:
                end_datetime = start_datetime + timedelta(days=1)
                end_date = end_datetime.strftime("%Y-%m-%d")
                end_time = "07:59:59"
            else:
                end_datetime = datetime.strptime(f"{end_date} {end_time}", "%Y-%m-%d %H:%M:%S")

            if start_datetime >= end_datetime:
                error_message = "Start Datetime must be before End Datetime."
                return render(request, self.template_name, {'error_message': error_message})

            # Define directory paths
            base_directory = '/home/icebox/itwatcher_api/nvr_videos/10.101.60.148'
            media_videos_directory = os.path.join(settings.MEDIA_ROOT, "videos")
            os.makedirs(media_videos_directory, exist_ok=True)

            # Construct download command
            command = f"python3 {script_path} {camera_ip} {start_date} {start_time} {end_date} {end_time} {channel}"
            if content_type:
                command += " -p"

            # Execute download command
            process = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                error_message = f"Download execution failed: {stderr.decode('utf-8')}"
                return render(request, self.template_name, {'error_message': error_message})

            # Iterate through all subdirectories
            for root, dirs, files in os.walk(base_directory):
                for sub_dir in dirs:
                    directory_path = os.path.join(root, sub_dir)
                    video_files = glob.glob(os.path.join(directory_path, "*.mp4"))

                    for video_file in video_files:
                        try:
                            # Extract and sanitize filename
                            original_filename = os.path.basename(video_file)
                            sanitized_filename = get_valid_filename(original_filename)
                            destination_path = os.path.abspath(os.path.join(media_videos_directory, sanitized_filename))

                            # Create placeholder entry in the database
                            video_instance = Video()
                            video_instance.save()  # Save the instance to generate an ID or primary key

                            # Move file to the destination path
                            shutil.move(video_file, destination_path)
                            if not os.path.exists(destination_path):
                                raise FileNotFoundError(f"Failed to move file: {video_file}")

                            # Associate the moved file with the Video instance
                            with open(destination_path, "rb") as f:
                                django_file = File(f)
                                video_instance.file.save(sanitized_filename, django_file)
                                video_instance.save()

                        except Exception as e:
                            print(f"Error processing video {video_file}: {e}")  # Log error details

                    # Delete subdirectory after processing
                    try:
                        shutil.rmtree(directory_path)
                    except Exception as e:
                        print(f"Failed to delete folder {directory_path}: {e}")

            # Delete untracked files
            self.delete_untracked_files(media_videos_directory)

        except Exception as e:
            error_message = f"An unexpected error occurred: {str(e)}"
            return render(request, self.template_name, {'error_message': error_message})

        return render(request, 'html_files/download_success.html', {})

    def delete_untracked_files(self, media_videos_directory):
        """Delete files not saved in the database."""
        saved_files = set(Video.objects.values_list('file', flat=True))

        for root, dirs, files in os.walk(media_videos_directory):
            for filename in files:
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, settings.MEDIA_ROOT)

                if relative_path not in saved_files:
                    try:
                        print(f"Deleting untracked file: {file_path}")
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")
                        
def success_page(request):
    return render(request, 'html_files/download_success.html')
           
class DownloadRequestDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = DownloadRequest.objects.all()
    serializer_class = DownloadRequestSerializer

def generate_report(request, log_id):
    # Fetch the ViolationLog instance
    violation_log_instance = ViolationLog.objects.get(pk=log_id)

    # Construct the filename with the unique identifier
    filename = f"Violation_Report_{log_id}.pdf"

    # Create a BytesIO buffer to store the PDF
    buffer = BytesIO()

    # Create a canvas
    c = canvas.Canvas(buffer, pagesize=A4)
    
    # Set up the logo
    logo_path = '/home/icebox/itwatcher_api/static/header.png'  # Update this with the actual path to your logo file
    logo_width = 600  # Adjust the width of the logo
    logo_height = 70   # Adjust the height of the logo
    logo_x = 0        # X position of the logo
    logo_y = 772       # Y position of the logo
    
    # Draw the logo image
    c.drawImage(logo_path, logo_x, logo_y, width=logo_width, height=logo_height)

    c.setFont("Helvetica-Bold", 12)
    # c.drawString(140, 800, "iTWatcher")
    c.drawString(30, 670, "VIOLATION REPORT")

    # Define the font and font size
    c.setFont("Helvetica", 11)
    c.drawString(30, 750, "Address: H-Building, Mindanao State University - General Santos City, General Santos City")
    c.drawString(30, 730, "Contact No.: 09171474280")
    c.drawString(30, 710, "Email: inteltraf.watcher@msugensan.edu.ph")
    
    # Write the table content
    c.drawString(170, 630, violation_log_instance.timestamp.strftime("%Y-%m-%d %H:%M:%S"))
    c.drawString(170, 610, str(violation_log_instance.video_file))
    c.drawString(170, 590, "Laurel Avenue")
    c.drawString(170, 570, str(violation_log_instance.vehicle_type))
    c.drawString(170, 550, str(violation_log_instance.vehicle_color))
    c.drawString(170, 530, str(violation_log_instance.plate_number))
    c.drawString(170, 510, str(violation_log_instance.violation))
    
    c.setFont("Helvetica-Bold", 12)

    # Write the table headers
    c.drawString(50, 630, "* Date and Time:")
    c.drawString(50, 610, "* Video Source:")
    c.drawString(50, 590, "* Location:")
    c.drawString(50, 570, "* Vehicle Type:")
    c.drawString(50, 550, "* Vehicle Color:")
    c.drawString(50, 530, "* Plate Number:")
    c.drawString(50, 510, "* Traffic Violation:")
    c.drawString(50, 470,"* License Plate Image:")
    c.drawString(50, 370,"* Screen Capture:")
    # Calculate the position and size of the plate image
    plate_image_width = 100
    plate_image_height = 50
    plate_image_x = 150
    plate_image_y = 450 - plate_image_height

    # Draw the plate image if available
    if violation_log_instance.plate_image:
        plate_image_path = violation_log_instance.plate_image.path
        plate_image = Image.open(plate_image_path)
        c.drawImage(plate_image_path, plate_image_x, plate_image_y, width=plate_image_width, height=plate_image_height)

    # Calculate the position and size of the frame image
    frame_image_width = 450
    frame_image_height = 280
    frame_image_x = 100
    frame_image_y = 350 - frame_image_height

    # Draw the frame image if available
    if violation_log_instance.frame_image:
        frame_image_path = violation_log_instance.frame_image.path
        frame_image = Image.open(frame_image_path)
        c.drawImage(frame_image_path, frame_image_x, frame_image_y, width=frame_image_width, height=frame_image_height)

    # Save the PDF
    c.showPage()
    c.save()

    # Go to the beginning of the buffer
    buffer.seek(0)

    # Create a Django response and return the PDF with the unique filename
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    response.write(buffer.getvalue())
    return response


def download_video_clip(request, log_id):
    log = get_object_or_404(ViolationLog, id=log_id)
    name = log.plate_number

    # Use only the first word of the violation
    first_word_of_violation = log.violation.split()[0]
    
    # Construct video file name with the first word of the violation
    video_file_with_prefix = f'{first_word_of_violation}_{log.video_file}'
    video_path = os.path.join(log.full_video_path, video_file_with_prefix)
    print(f"Constructed video path: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"Video file not found at {video_path}")
        return render(request, "html_files/vid_warning.html", {
            "message": "The video clip is still being processed. Please try downloading it later."
        })
    
    try:
        # Check if log contains a valid frame number
        if log.frame_number is None:
            raise Http404("Frame number not available in log.")
        
        # Frame rate (FPS) of the video - you should ideally get this from the video metadata
        fps = 30  # Assuming 30 FPS, adjust this to match the actual FPS of your video
        
        # Calculate the start and end frame numbers (10 frames before and after the violation frame)
        violation_frame = log.frame_number
        start_frame = max(0, violation_frame - 50)  # Ensure frame is not negative
        end_frame = violation_frame + 50
        
        print(f"Violation frame: {violation_frame}, Start frame: {start_frame}, End frame: {end_frame}")
        
        # Convert frame numbers to seconds
        start_time = start_frame / fps
        end_time = end_frame / fps
        
        print(f"Start time: {start_time}, End time: {end_time}")
        
        # Ensure that start_time is always less than end_time
        if start_time >= end_time:
            raise Http404("Invalid time range for video clip.")

        # Define a temporary file path for the clipped video
        temp_file_path = f"/tmp/clip_{log_id}.mp4"

        # Perform the clipping using FFmpeg
        ffmpeg.input(video_path, ss=start_time, to=end_time).output(temp_file_path, codec="libx264").run()
        print(f"Clip created successfully: {temp_file_path}")

        # Serve the video for download
        with open(temp_file_path, "rb") as f:
            response = HttpResponse(f.read(), content_type="video/mp4")
            response["Content-Disposition"] = f'attachment; filename="violation_clip_{name}-{log_id}.mp4"'

        # Clean up temporary file
        os.remove(temp_file_path)
        return response

    except Exception as e:
        # print(f"Error processing video {video_path}: {e}")
        # raise Http404(f"Error processing video: {e}")
        return render(request, "html_files/vid_warning.html", {
            "message": "The video clip is still being processed. Please try downloading it later."
        })
