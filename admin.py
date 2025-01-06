from django.contrib import admin
# from .models import Video, PlateLog, CountLog, VehicleLog, ColorLog, NVRVideo, OutputVideo, SwervingVideo, BlockingVideo, LPRVideo, ColorVideo, ViolationLog
from .models import Video, PlateLog, CountLog, VehicleLog, ColorLog, ViolationLog

# Register your models here.
admin.site.register(Video)
admin.site.register(CountLog)
admin.site.register(VehicleLog)
admin.site.register(ColorLog)

class PlateLogAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'video_file','display_frame_image', 'display_plate_image','plate_number' )  # Include 'display_plate_image' here
admin.site.register(PlateLog, PlateLogAdmin)

class ViolationLogAdmin(admin.ModelAdmin):
    list_display = ('timestamp_date', 'timestamp_time','video_file','display_vehicle_image', 'vehicle_type','vehicle_color', 'display_plate_image', 'plate_number', 'display_frame_image',  'violation' )  # Include 'display_plate_image' here
admin.site.register(ViolationLog, ViolationLogAdmin)
