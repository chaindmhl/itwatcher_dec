from tracking.models import VehicleLog

# Delete all VehicleLog entries
VehicleLog.objects.all().delete()