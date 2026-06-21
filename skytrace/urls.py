from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from app import views

urlpatterns = [
    path('', views.index, name='index'),
    path('auth/login/', views.login_view, name='login'),
    path('auth/logout/', views.logout_view, name='logout'),
    path('api/upload/', views.upload_log, name='upload_log'),
    path('api/logs/', views.get_logs, name='get_logs'),
    path('api/dashboard/', views.dashboard_data, name='dashboard_data'),
    path('api/view-log/<int:log_id>/', views.view_log_content, name='view_log_content'),
    path('api/analysis/<int:log_id>/', views.analyze_log, name='analyze_log'),
    path('api/report/<int:report_id>/', views.get_report, name='get_report'),
    path('api/verify/<int:report_id>/', views.verify_report, name='verify_report'),
    path('api/download/<int:report_id>/<str:fmt>/', views.download_report, name='download_report'),
    path('api/graphs/<int:log_id>/', views.analyze_graphs, name='analyze_graphs'),
    path('api/delete-log/<int:log_id>/', views.delete_log, name='delete_log'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
