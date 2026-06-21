# SkyTrace – Log Analysis System

A full-stack Django web application for ML-based log analysis.

## Setup & Run

```bash
# Install dependencies
pip install django reportlab

# Apply migrations
python manage.py migrate

# Run server
python manage.py runserver
```

Then open http://127.0.0.1:8000 in your browser.

## Features
- Auto-register / login via email
- Upload .txt / .log files
- GRU-based anomaly detection (simulated)
- Dashboard with live stats
- Analysis page with critical issues + anomaly scores
- Report verification with notes
- Download reports as PDF / JSON / TXT

## Tech Stack
- Backend: Django 4+ / SQLite
- Frontend: HTML, CSS, JavaScript (same UI as original)
- Reports: ReportLab (PDF)
