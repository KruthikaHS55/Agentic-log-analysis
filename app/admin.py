from django.contrib import admin
from .models import LogFile, AnalysisReport

@admin.register(LogFile)
class LogFileAdmin(admin.ModelAdmin):
    list_display = ['filename', 'user', 'status', 'uploaded_at', 'file_size']

@admin.register(AnalysisReport)
class AnalysisReportAdmin(admin.ModelAdmin):
    list_display = ['id', 'log_file', 'user', 'status', 'generated_at', 'verified_by']
