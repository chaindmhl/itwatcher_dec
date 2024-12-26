from rest_framework import serializers
import os
from subprocess import run
import shutil
import tempfile
from django.contrib.auth.models import User

from django.conf import settings
from tracking.models import (
    Video, 
    CountLog,
    VehicleLog,
    DownloadRequest,
)


class VideoSerializers(serializers.ModelSerializer):
    file = serializers.FileField(max_length=500)

    class Meta:
        model = Video
        fields = ["id", "file"]


class CountLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = CountLog
        fields = '__all__'
    
class VehicleLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = VehicleLog
        fields = '__all__'

class DownloadRequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = DownloadRequest
        fields = '__all__'

# class ProcessVideoSerializers(serializers.Serializer):
#     #video_path = serializers.CharField(required=False)
#     livestream_url = serializers.CharField(required=False)
#     video_path = serializers.PrimaryKeyRelatedField(queryset=Video.objects.all(), required=False)

#     def to_representation(self, instance):
#         data = super().to_representation(instance)
#         if data.get('video_path'):
#             data['video_name'] = data['video_path'].name
#         return data
    
#     def validate(self, attrs):
#         video_path = attrs.get('video_path')
#         livestream_url = attrs.get('livestream_url')

#         if not video_path and not livestream_url:
#             raise serializers.ValidationError('Either video path or camera URL must be provided.')

#         if video_path and livestream_url:
#             raise serializers.ValidationError('Only one of video path or camera URL can be provided.')

#         if video_path:
#             # check if the file path exists
#             video_path_str = str(video_path.file.path)
#             if not os.path.exists(video_path_str):
#                 raise serializers.ValidationError(f'Video file path {video_path_str} does not exist.')
#             attrs['video'] = video_path_str

#         elif livestream_url:
#             livestream_url_str = livestream_url
#             attrs['video'] = livestream_url_str
#         return attrs

#     def create(self, validated_data):
#         video = validated_data["video"]
#         return video

#     def create(self, validated_data):
#         video_files = validated_data.get("video_files", [])
#         livestream_url = validated_data.get("video")

#         # Return video files or livestream URL based on input
#         return {"video_files": video_files} if video_files else {"livestream_url": livestream_url}

class ProcessVideoSerializers(serializers.Serializer):
    livestream_url = serializers.CharField(required=False, allow_blank=True)
    video_path = serializers.PrimaryKeyRelatedField(queryset=Video.objects.all(), required=False)
    video_files = serializers.ListField(
        child=serializers.CharField(), required=False
    )  # Handles directory paths

    

    def to_representation(self, instance):
        data = super().to_representation(instance)
        if data.get('video_path'):
            data['video_name'] = data['video_path'].name
        return data

    def validate(self, attrs):
        video_path = attrs.get('video_path')
        livestream_url = attrs.get('livestream_url')
        video_files = attrs.get('video_files')

        print(f"video_path: {video_path}")
        print(f"livestream_url: {livestream_url}")
        print(f"video_files: {video_files}")

        # Ensure at least one input is provided
        if not video_path and not livestream_url and not video_files:
            raise serializers.ValidationError(
                "Either video path, video files, or camera URL must be provided."
            )

        # Prevent multiple inputs being provided simultaneously
        if (video_path and livestream_url) or (video_path and video_files) or (livestream_url and video_files):
            raise serializers.ValidationError(
                "Only one of video path, video files, or camera URL can be provided."
            )

        if video_path:
            # Check if the path is a directory or a file
            video_path_str = str(video_path.file.path)

            if os.path.isdir(video_path_str):
                # Handle directory case
                video_files_list = [
                    os.path.join(video_path_str, f)
                    for f in os.listdir(video_path_str)
                    if f.lower().endswith(('.mp4', '.avi', '.mov'))
                ]
                if not video_files_list:
                    raise serializers.ValidationError(
                        f"No video files found in directory: {video_path_str}"
                    )
                attrs['video_files'] = video_files_list  # Add files to attrs
                attrs['video'] = video_path_str  # Add directory path
            elif os.path.isfile(video_path_str):
                attrs['video'] = video_path_str  # Add file path
            else:
                raise serializers.ValidationError(
                    f"Video path {video_path_str} does not exist."
                )

        if livestream_url:
            # Optionally validate URL format here
            attrs['video'] = livestream_url  # Add URL


        return attrs


class LPRSerializers(serializers.Serializer):
    camera_url = serializers.CharField(required=False)
    video_path = serializers.PrimaryKeyRelatedField(queryset=Video.objects.all(),required=False)

    def validate(self, attrs):
        video_path = attrs.get('video_path')
        camera_url = attrs.get('camera_url')

        if not video_path and not camera_url:
            raise serializers.ValidationError('Either video path or camera URL must be provided.')

        if video_path and camera_url:
            raise serializers.ValidationError('Only one of video path or camera URL can be provided.')

        if video_path:
            # check if the file path exists
            video_path_str = str(video_path.file.path)
            if not os.path.exists(video_path_str):
                raise serializers.ValidationError(f'Video file path {video_path_str} does not exist.')
            attrs['input_video'] = video_path_str

        if camera_url:
            attrs['input_video'] = camera_url

        return attrs

    def create(self, validated_data):
        INPUT_VIDEO = validated_data["input_video"]
        output_dir = "/home/itwatcher/tricycle_copy/lpr_output"
        with tempfile.TemporaryDirectory() as tmpdir:
            command = ["python", "lpr.py", f"--input_video={validated_data['input_video']}", f"--output_video={os.path.join(tmpdir, 'output.mp4')}"]
            result = run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Error running lpr.py: {result.stderr.strip()}")
            # Copy the output video to the output directory
            output_file = os.path.join(output_dir, "output.mp4")
            shutil.copyfile(os.path.join(tmpdir, "output.mp4"), output_file)
        return output_file
