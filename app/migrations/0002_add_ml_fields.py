from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("app", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="analysisreport",
            name="model_name",
            field=models.CharField(default="GRU", max_length=50),
        ),
        migrations.AddField(
            model_name="analysisreport",
            name="duplicate_count",
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name="analysisreport",
            name="event_count",
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name="analysisreport",
            name="all_entries",
            field=models.TextField(default="[]"),
        ),
    ]
