from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from .views import DownloadRequestListCreateView, DownloadRequestDetailView, success_page
from tracking.views import (
    VideoUploadViewSet,
    ProcessTrikeViewSet,
    CatchAllViewSet,
    LPRTrikeViewSet,
    LPRAllViewSet,
    CombiViewSet,
    LPRCombiViewSet,
    # ColorViewSet,
    RedLightViewSet,
    BlockingViewSet,
    SpeedViewSet,
    MyView,
    UploadView,
    PlateView,
    FrameView,
    FrameViolationView,
    MapView,
    ColorView,
    CountLogListView,
    ViolationView,
    TrikeVehicleLogListView,
    VehicleLogListView,
    TricycleCountGraphView,
    VehicleCountGraphView,
    SignupView,
    CustomLoginView,
    DarknetTrainView,
    LPRView,
    TrackCountView,
    ColorRecognitionView,
    VioDetectionView,
    UploadView,
    generate_report,
    update_plate_number
)

router = DefaultRouter()
router.register("tracking/video", VideoUploadViewSet, basename="tracking-video")
# router.register("tracking/color", ColorViewSet, basename="tracking-color")
router.register("tracking/tric", ProcessTrikeViewSet, basename="tracking-tric")
router.register("tracking/catchall", CatchAllViewSet, basename="tracking-catchall")
router.register("tracking/combi", CombiViewSet, basename="tracking-combi")
router.register("tracking/lpr_trike", LPRTrikeViewSet, basename="LPR-tric")
# router.register("tracking/lpr-cam", LPRTrikeViewSet, basename="LPR-tric-cam")
router.register("tracking/lpr_all", LPRAllViewSet, basename="LPR-All_Vehicle")
router.register("tracking/lpr_combi", LPRCombiViewSet, basename="LPR-combi")
router.register("tracking/redlight", RedLightViewSet, basename="tracking-redlight")
router.register("tracking/blocking", BlockingViewSet, basename="tracking-blocking")
router.register("tracking/speeding", SpeedViewSet, basename="tracking-speeding")

urlpatterns = [
    path('', include(router.urls)),
    path('my-url/', MyView.as_view(), name='my-view'),
    path('download-requests/', DownloadRequestListCreateView.as_view(), name='downloadrequest-list-create'),
    path('upload-requests/', UploadView.as_view(), name='upload-video'),
    path('download-requests/<int:pk>/', DownloadRequestDetailView.as_view(), name='downloadrequest-detail'),
    path('success/', success_page, name='success-page'),
    path('display_plates/', PlateView.as_view(), name='display_plates'),
    path('display_color/', ColorView.as_view(), name='display_color'),
    path('violation_list/', ViolationView.as_view(), name='violation'),
    path('view_frame/<int:log_id>/', FrameView.view_frame, name='view_frame'),
    path('view_violationframe/<int:log_id>/', FrameViolationView.view_violationframe, name='view_violationframe'),
    path('view_camera_map/', MapView.view_camera_map, name='view_camera_map'),
    path('count_logs/', CountLogListView.as_view(), name='count_log_list'),
    path('vehicle_logs/', VehicleLogListView.as_view(), name='vehicle_log_list'),
    path('trikeall_logs/', TrikeVehicleLogListView.as_view(), name='trikeall_log_list'),
    path('tricycle_count_graph/<int:log_id>/', TricycleCountGraphView.as_view(), name='tricycle_count_graph'),
    path('vehicle_count_graph/<str:log_date>/<int:log_id>/', VehicleCountGraphView.as_view(), name='vehicle_count_graph'),
    path('signup/', SignupView.as_view(), name='signup'),
    path('login/', CustomLoginView.as_view(), name='login'),
    path('train/', DarknetTrainView.as_view(), name='train'),
    path('lpr/', LPRView.as_view(), name='lpr-view'),
    path('track/', TrackCountView.as_view(), name='track-count'),
    path('color/', ColorRecognitionView.as_view(), name='color'),
    path('violation/', VioDetectionView.as_view(), name='viodetection'),
    path('upload-video/', UploadView.as_view(), name='upload-video'),
    path('violation_report/<int:log_id>/', generate_report, name='generate_report'),
    path('update-plate-number/', update_plate_number, name='update_plate_number'),
    path('control-traffic-light/', RedLightViewSet.as_view({'post': 'control_traffic_light'}), name='control-traffic-light'),
    path('download_clip/<int:log_id>/', views.download_video_clip, name='download_clip'),  
    

] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
