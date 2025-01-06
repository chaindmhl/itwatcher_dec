from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.utils.html import format_html
from PIL import Image
from datetime import datetime, timedelta
import os
from django.core.files import File
from tracking.utils.load_config import load_config


# Load the configuration once
_, processed_defaults, last_processed_defaults = load_config()


class Video(models.Model):
    file = models.FileField(upload_to="videos/", blank=True)

    # Explicitly define the fields for processed status
    processed_tc_trike = models.BooleanField(default=processed_defaults.get("processed_tc_trike", False))
    processed_tc_catchall = models.BooleanField(default=processed_defaults.get("processed_tc_catchall", False))
    processed_tc_itwatcher = models.BooleanField(default=processed_defaults.get("processed_tc_itwatcher", False))
    processed_lpr_trike = models.BooleanField(default=processed_defaults.get("processed_lpr_trike", False))
    processed_lpr_catchall = models.BooleanField(default=processed_defaults.get("processed_lpr_catchall", False))
    processed_lpr_itwatcher = models.BooleanField(default=processed_defaults.get("processed_lpr_itwatcher", False))
    processed_brl = models.BooleanField(default=processed_defaults.get("processed_brl", False))
    processed_bpl = models.BooleanField(default=processed_defaults.get("processed_bpl", False))
    processed_os = models.BooleanField(default=processed_defaults.get("processed_os", False))

    # Explicitly define the fields for last processed timestamps
    last_processed_tc_trike = models.DateTimeField(null=True, blank=True, default=last_processed_defaults.get("last_processed_tc_trike"))
    last_processed_tc_catchall = models.DateTimeField(null=True, blank=True, default=last_processed_defaults.get("last_processed_tc_catchall"))
    last_processed_tc_itwatcher = models.DateTimeField(null=True, blank=True, default=last_processed_defaults.get("last_processed_tc_itwatcher"))
    last_processed_lpr_trike = models.DateTimeField(null=True, blank=True, default=last_processed_defaults.get("last_processed_lpr_trike"))
    last_processed_lpr_catchall = models.DateTimeField(null=True, blank=True, default=last_processed_defaults.get("last_processed_lpr_catchall"))
    last_processed_lpr_itwatcher = models.DateTimeField(null=True, blank=True, default=last_processed_defaults.get("last_processed_lpr_itwatcher"))
    last_processed_brl = models.DateTimeField(null=True, blank=True, default=last_processed_defaults.get("last_processed_brl"))
    last_processed_bpl = models.DateTimeField(null=True, blank=True, default=last_processed_defaults.get("last_processed_bpl"))
    last_processed_os = models.DateTimeField(null=True, blank=True, default=last_processed_defaults.get("last_processed_os"))

class BasenameField(models.CharField):
    def to_python(self, value):
        value = super().to_python(value)
        if value is not None:
            value = os.path.basename(value)
        return value

    def from_db_value(self, value, expression, connection):
        return self.to_python(value)

class DownloadRequest(models.Model):
    start_datetime = models.DateTimeField(default=None, blank=True, null=True)
    end_datetime = models.DateTimeField(default=None, blank=True, null=True)
    channel = models.CharField(max_length=100)
    directory_path = models.CharField(max_length=255, default=None, blank=True, null=True)
    timestamp = models.DateTimeField(default=timezone.now)

    
class CountLog(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    filename = models.CharField(max_length=255, null=True)
    total_count = models.PositiveIntegerField(null=True)
    hwy_count = models.PositiveIntegerField(null=True)
    msu_count = models.PositiveIntegerField(null=True)
    sm_count = models.PositiveIntegerField(null=True)
    oval_count = models.PositiveIntegerField(null=True)
    class_counts = models.JSONField(default=dict)
    class_counter = models.JSONField(default=dict)
    
    def __str__(self):
        return f"CountLog - {self.timestamp}"

class VehicleLog(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    date = models.DateField(default=timezone.now)
    start_time = models.TimeField(default='00:00:00')
    end_time = models.TimeField(default='00:00:00')
    filename = BasenameField(max_length=255, null=True)
    total_count = models.PositiveIntegerField(null=True)
    hwy_count = models.PositiveIntegerField(null=True)
    msu_count = models.PositiveIntegerField(null=True)
    sm_count = models.PositiveIntegerField(null=True)
    oval_count = models.PositiveIntegerField(null=True)
    class_counts = models.JSONField(default=dict)
    class_counter = models.JSONField(default=dict)
    
    @property
    def log_date(self):
        return self.timestamp.date()
    
    def __str__(self):
        return f"Vehicle Log - Date: {self.date}, Time Interval: {self.start_time} - {self.end_time}"

class PlateLog(models.Model):
    timestamp = models.DateTimeField(default=timezone.now)
    filename = models.CharField(max_length=255, null=True)
    video_file = BasenameField(max_length=255, null=True)
    plate_number = models.CharField(max_length=255, null=True)
    plate_image = models.ImageField(upload_to='plate_images/', null=True)
    warped_image = models.ImageField(upload_to='warped_images/', null=True)
    frame_image = models.ImageField(upload_to='frame_images/', null=True)

    def __str__(self):
        return f"PlateLog for {self.timestamp}"
    
    def display_plate_image(self):

        if self.plate_image:
            # Open the image using Pillow
            image = Image.open(self.plate_image.path)
            
            # Resize the image while maintaining the aspect ratio
            max_width = 200  # Set the maximum width for display
            width_percent = (max_width / float(image.size[0]))
            new_height = int(float(image.size[1]) * float(width_percent))
            #resized_image = image.resize((max_width, new_height), Image.ANTIALIAS)
            
            # Use format_html to ensure HTML is not escaped
            return format_html('<img src="{}" width="{}" height="{}" />', self.plate_image.url, max_width, new_height)
        else:
            return format_html('<img src="{}" width="{}" height="{}" />', '/static/placeholder.png', 200, 200)
    
    def display_warped_image(self):

        if self.warped_image:
            # Open the image using Pillow
            image = Image.open(self.warped_image.path)
            
            # Resize the image while maintaining the aspect ratio
            max_width = 200  # Set the maximum width for display
            width_percent = (max_width / float(image.size[0]))
            new_height = int(float(image.size[1]) * float(width_percent))
            #resized_image = image.resize((max_width, new_height), Image.ANTIALIAS)
            
            # Use format_html to ensure HTML is not escaped
            return format_html('<img src="{}" width="{}" height="{}" />', self.warped_image.url, max_width, new_height)
        else:
            return format_html('<img src="{}" width="{}" height="{}" />', '/static/placeholder.png', 200, 200)
        
    def display_frame_image(self):

        if self.frame_image:
            # Open the image using Pillow
            image = Image.open(self.frame_image.path)
            
            # Resize the image while maintaining the aspect ratio
            max_width = 200  # Set the maximum width for display
            width_percent = (max_width / float(image.size[0]))
            new_height = int(float(image.size[1]) * float(width_percent))
            #resized_image = image.resize((max_width, new_height), Image.ANTIALIAS)

            # Use format_html to ensure HTML is not escaped
            return format_html('<img src="{}" width="{}" height="{}" />', self.frame_image.url, max_width, new_height)
        else:
            return format_html('<img src="{}" width="{}" height="{}" />', '/static/placeholder.png', 200, 200)

    display_plate_image.short_description = "Plate Image"
    display_warped_image.short_description = "Warped Plate Image"
    display_frame_image.short_description = "Frame Image"

class ColorLog(models.Model):
    timestamp = models.DateTimeField(default=timezone.now)
    filename = models.CharField(max_length=255, null=True)
    vehicle_image = models.ImageField(upload_to='vehicle_images/', null=True)
    color = models.CharField(max_length=255, null=True)
    frame_image = models.ImageField(upload_to='frame_images/', null=True)

    def __str__(self):
        return f"ColorLog for {self.timestamp}"
    
    def display_vehicle_image(self):

        if self.vehicle_image:
            # Open the image using Pillow
            image = Image.open(self.vehicle_image.path)
            
            # Resize the image while maintaining the aspect ratio
            max_width = 200  # Set the maximum width for display
            width_percent = (max_width / float(image.size[0]))
            new_height = int(float(image.size[1]) * float(width_percent))
            #resized_image = image.resize((max_width, new_height), Image.ANTIALIAS)
            
            # Use format_html to ensure HTML is not escaped
            return format_html('<img src="{}" width="{}" height="{}" />', self.vehicle_image.url, max_width, new_height)
        else:
            return format_html('<img src="{}" width="{}" height="{}" />', '/static/placeholder.png', 200, 200)
        
    def display_frame_image(self):

        if self.frame_image:
            # Open the image using Pillow
            image = Image.open(self.frame_image.path)
            
            # Resize the image while maintaining the aspect ratio
            max_width = 200  # Set the maximum width for display
            width_percent = (max_width / float(image.size[0]))
            new_height = int(float(image.size[1]) * float(width_percent))
            #resized_image = image.resize((max_width, new_height), Image.ANTIALIAS)

            # Use format_html to ensure HTML is not escaped
            return format_html('<img src="{}" width="{}" height="{}" />', self.frame_image.url, max_width, new_height)
        else:
            return format_html('<img src="{}" width="{}" height="{}" />', '/static/placeholder.png', 200, 200)

    display_vehicle_image.short_description = "Vehicle Image"
    display_frame_image.short_description = "Frame Image"

def get_time_with_offset():
    now = timezone.now()
    # Add 8 hours to the current time
    return (now + timedelta(hours=8)).time()

class ViolationLog(models.Model):
    # timestamp = models.DateTimeField(default=timezone.now)
    timestamp_date = models.DateField(default=timezone.now)  # Stores only the date
    timestamp_time = models.TimeField(default=get_time_with_offset)
    video_file = BasenameField(max_length=255, null=True)
    full_video_path = models.CharField(max_length=1024, null=True, blank=True)  # New field for video
    frame_number = models.IntegerField(null=False, default=0)  # Add a frame_number field
    vehicle_image = models.ImageField(upload_to='vehicle_images/', null=True)
    vehicle_type =  models.CharField(max_length=255, null=True)
    vehicle_color = models.CharField(max_length=255, null=True)
    plate_image = models.ImageField(upload_to='plate_images/', null=True)
    plate_number = models.CharField(max_length=255, null=True)
    violation = models.CharField(max_length=255, null=True)
    frame_image = models.ImageField(upload_to='frame_images/', null=True)

    def __str__(self):
        return f"Violation Log for {self.timestamp_date}-{self.timestamp_time}"
    
    def display_plate_image(self):

        if self.plate_image:
            # Open the image using Pillow
            image = Image.open(self.plate_image.path)
            
            # Resize the image while maintaining the aspect ratio
            max_width = 200  # Set the maximum width for display
            width_percent = (max_width / float(image.size[0]))
            new_height = int(float(image.size[1]) * float(width_percent))
            #resized_image = image.resize((max_width, new_height), Image.ANTIALIAS)
            
            # Use format_html to ensure HTML is not escaped
            return format_html('<img src="{}" width="{}" height="{}" />', self.plate_image.url, max_width, new_height)
        else:
            return format_html('<img src="{}" width="{}" height="{}" />', '/static/placeholder.png', 200, 200)
        
    def display_vehicle_image(self):

        if self.vehicle_image:
            # Open the image using Pillow
            image = Image.open(self.vehicle_image.path)
            
            # Resize the image while maintaining the aspect ratio
            max_width = 200  # Set the maximum width for display
            width_percent = (max_width / float(image.size[0]))
            new_height = int(float(image.size[1]) * float(width_percent))
            #resized_image = image.resize((max_width, new_height), Image.ANTIALIAS)
            
            # Use format_html to ensure HTML is not escaped
            return format_html('<img src="{}" width="{}" height="{}" />', self.vehicle_image.url, max_width, new_height)
        else:
            return format_html('<img src="{}" width="{}" height="{}" />', '/static/placeholder.png', 200, 200)
        
    def display_frame_image(self):

        if self.frame_image:
            # Open the image using Pillow
            image = Image.open(self.frame_image.path)
            
            # Resize the image while maintaining the aspect ratio
            max_width = 200  # Set the maximum width for display
            width_percent = (max_width / float(image.size[0]))
            new_height = int(float(image.size[1]) * float(width_percent))
            #resized_image = image.resize((max_width, new_height), Image.ANTIALIAS)

            # Use format_html to ensure HTML is not escaped
            return format_html('<img src="{}" width="{}" height="{}" />', self.frame_image.url, max_width, new_height)
        else:
            return format_html('<img src="{}" width="{}" height="{}" />', '/static/placeholder.png', 200, 200)

    display_plate_image.short_description = "Plate Image"
    display_vehicle_image.short_description = "Vehicle Image"
    display_frame_image.short_description = "Frame Image"
