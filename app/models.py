from django.db import models
from django.contrib.auth.models import User


class LogFile(models.Model):
    STATUS_CHOICES = [
        ('uploaded', 'Uploaded'),
        ('analyzed', 'Analyzed'),
        ('verified', 'Verified'),
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='logs/')
    filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='uploaded')
    file_size = models.BigIntegerField(default=0)

    class Meta:
        unique_together = ('user', 'filename')

    def __str__(self):
        return self.filename

    def size_display(self):
        size = self.file_size
        if size >= 1024 * 1024:
            return f"{size / (1024*1024):.2f} MB"
        elif size >= 1024:
            return f"{size / 1024:.2f} KB"
        return f"{size} B"


class AnalysisReport(models.Model):
    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('verified', 'Verified'),
    ]
    log_file = models.ForeignKey(LogFile, on_delete=models.CASCADE, related_name='reports')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    generated_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='draft')
    model_name = models.CharField(max_length=50, default='GRU')

    # Analysis results
    total_lines = models.IntegerField(default=0)
    error_count = models.IntegerField(default=0)
    warning_count = models.IntegerField(default=0)
    critical_count = models.IntegerField(default=0)
    info_count = models.IntegerField(default=0)
    debug_count = models.IntegerField(default=0)
    anomaly_count = models.IntegerField(default=0)
    duplicate_count = models.IntegerField(default=0)
    event_count = models.IntegerField(default=0)

    critical_lines = models.TextField(default='[]')
    anomaly_lines = models.TextField(default='[]')
    all_entries = models.TextField(default='[]')

    # Verification
    verified_by = models.CharField(max_length=100, blank=True)
    verified_at = models.DateTimeField(null=True, blank=True)
    verification_notes = models.TextField(blank=True)

    def __str__(self):
        return f"Report #{self.id} for {self.log_file.filename}"
