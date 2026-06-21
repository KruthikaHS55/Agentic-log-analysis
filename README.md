# HPE SkyTrace Alpha 1.0
### AI-Powered Log Anomaly Detection System

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-4.x-green.svg)](https://www.djangoproject.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SkyTrace is a full-stack web application that automatically detects anomalies in HPE FabricOS switch logs using a multi-model ML pipeline. It classifies every log line as **Normal**, **Warning**, or **Critical** and compares the performance of 6 different ML algorithms.

---

## Features

- Upload `.txt` / `.log` switch log files
- Automatic classification: Normal / Warning / Critical
- 6 ML algorithms with side-by-side comparison
- Accuracy, Precision, Recall, F1, ROC-AUC metrics per model
- Duplicate event detection with 5-minute sliding window
- Cross-validation layer that catches mislabeled INFO-level critical events
- Downloadable PDF reports
- User authentication (auto-register on first login)
- Dashboard with live stats

---

## ML Models Included

| Model | Type |
|-------|------|
| GRU | Gated Recurrent Unit |
| LSTM | Long Short-Term Memory |
| Isolation Forest | Hyperparameter-tuned (99.76% accuracy) |
| GRU Autoencoder | Reconstruction-based anomaly detection |
| LSTM + GRU | Hybrid sequential model |
| Isolation Forest + GRU AE | Novel hybrid (tree isolation + autoencoder) |

> All models are implemented in pure NumPy — no TensorFlow or PyTorch required.

---

## Tech Stack

- **Backend:** Python 3.11, Django 4.x
- **Database:** SQLite
- **ML Engine:** NumPy (custom implementations)
- **Frontend:** HTML5, CSS3, JavaScript, Chart.js
- **Reports:** ReportLab (PDF generation)
- **Metrics:** scikit-learn (ROC/AUC only)

---

## Prerequisites

Make sure you have the following installed:

- Python 3.9 or higher → [Download](https://www.python.org/downloads/)
- pip (comes with Python)
- Git → [Download](https://git-scm.com/)

Check your versions:
```bash
python3 --version
pip3 --version
git --version
```

---

## Installation & Setup

### Step 1 — Clone the Repository

```bash
git clone https://github.com/KruthikaHS55/Agentic-log-analysis.git
cd Agentic-log-analysis
```

### Step 2 — Create a Virtual Environment (Recommended)

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install django
pip install numpy
pip install scikit-learn
pip install reportlab
```

Or install all at once:
```bash
pip install django numpy scikit-learn reportlab
```

### Step 4 — Apply Database Migrations

```bash
python manage.py migrate
```

### Step 5 — Run the Server

```bash
python manage.py runserver
```

### Step 6 — Open in Browser

```
http://127.0.0.1:8000
```

---

## First Time Login

SkyTrace auto-creates your account on first login — no separate registration needed.

1. Open `http://127.0.0.1:8000`
2. Enter any email and password
3. Your account is created automatically
4. You are logged in and ready to use the system

---

## How to Use

### Upload a Log File
1. Click **Upload Log** in the navigation bar
2. Select a `.txt` or `.log` file from your system
3. Click Upload

### Run Analysis
1. Go to **View Log Files**
2. Select an algorithm from the dropdown (e.g. Isolation Forest)
3. Click **Analyze**
4. View results — every line classified with method and score

### View Graphs
1. Click **Graphs** next to any analyzed file
2. See accuracy, precision, recall, F1, confusion matrix, ROC curve, score distribution

### Compare Algorithms
1. Click **Compare** next to any analyzed file
2. Check the algorithms you want to compare (select one at a time, click Add to Comparison)
3. Add at least 2 entries
4. Click **Compare All Entries**
5. See side-by-side performance charts

### Download Report
1. Click **View** on any analyzed file
2. Click **Download PDF** to get a full analysis report

### Verify Report
1. Go to **Verify Report** in the navigation
2. Add verification notes and sign off on the analysis

---

## Project Structure

```
Agentic-log-analysis/
│
├── app/
│   ├── analyze_logs.py      # Core ML pipeline (Rule Engine + 6 ML models)
│   ├── views.py             # Django views + ML model implementations for UI
│   ├── models.py            # Database models (LogFile, AnalysisReport)
│   ├── admin.py             # Django admin config
│   └── migrations/          # Database migration files
│
├── skytrace/
│   ├── settings.py          # Django settings
│   ├── urls.py              # URL routing
│   └── wsgi.py              # WSGI config
│
├── templates/
│   └── index.html           # Full frontend (single-page app)
│
├── static/
│   └── logo.svg             # HPE SkyTrace logo
│
├── manage.py                # Django management script
├── README.md                # This file
└── .gitignore
```

---

## Sample Log Format Supported

```
2026/03/09-04:27:47 (GMT), [BL-1090], 2914, CHASSIS | PORT 0/24, ERROR, sw01, Optical module I2C error
2026/01/08-06:20:25 (GMT), [SEC-3076], 2076, CHASSIS, INFO, sw01, SSH Session establishment failed
2026/02/02-11:46:27 (GMT), [ZONE-1076], 2451, FID 128, WARNING, sw01, Zone fabric lock cancelled
```

The system also handles free-text log lines (no structured format required).

---

## Detection Results on Sample Data

Tested on `Errdump.txt` (2,059 lines from a real HPE switch):

| Metric | Value |
|--------|-------|
| Total lines | 2,059 |
| Normal | 2,036 (98.8%) |
| Warning | 16 (0.8%) |
| Critical | 7 (0.3%) |
| Isolation Forest Accuracy | **99.76%** |
| Recall (critical events) | **100%** |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'django'`**
```bash
pip install django
```

**`sqlite3.OperationalError: disk I/O error`**
```bash
rm db.sqlite3
python manage.py migrate
```

**`Port 8000 already in use`**
```bash
python manage.py runserver 8080
```
Then open `http://127.0.0.1:8080`

**`No module named 'reportlab'`**
```bash
pip install reportlab
```

---

## Authors

- **Kruthika H S** — Developer
- Built as part of HPE CPP 2026 Project

---

## Acknowledgements

- HPE (Hewlett Packard Enterprise) — Project domain and log data
- Django Project — Web framework
- NumPy — Numerical computing
- Chart.js — Frontend visualizations
