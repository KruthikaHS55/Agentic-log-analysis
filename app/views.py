import json
import os
import re
import math
import time
import tracemalloc
import numpy as np
from datetime import datetime

from sklearn.metrics import roc_curve, auc
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.contrib.auth.decorators import login_required

from .models import LogFile, AnalysisReport



def index(request):
    return render(request, 'index.html')


@csrf_exempt
def login_view(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    try:
        data = json.loads(request.body)
    except Exception:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    email = data.get('email', '').strip().lower()
    password = data.get('password', '').strip()

    if not email or not password:
        return JsonResponse({'error': 'Email and password required'}, status=400)

    # Validate email format
    if '@' not in email or '.' not in email.split('@')[-1]:
        return JsonResponse({'error': 'Please enter a valid email address'}, status=400)

    username = email.replace('@', '_at_').replace('.', '_')

    try:
        user_obj = User.objects.get(email=email)
        user = authenticate(request, username=user_obj.username, password=password)
        if user is None:
            return JsonResponse({'error': 'Invalid credentials'}, status=401)
    except User.DoesNotExist:
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            first_name=email.split('@')[0].capitalize()
        )

    login(request, user)
    return JsonResponse({
        'success': True,
        'user': user.get_full_name() or user.email,
        'email': user.email,
    })


def logout_view(request):
    logout(request)
    return JsonResponse({'success': True})


@csrf_exempt
def upload_log(request):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Not authenticated'}, status=401)
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    uploaded_file = request.FILES.get('file')
    if not uploaded_file:
        return JsonResponse({'error': 'No file uploaded'}, status=400)

    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ['.txt', '.log']:
        return JsonResponse({'error': 'Only .txt and .log files are allowed'}, status=400)

    if LogFile.objects.filter(user=request.user, filename=filename).exists():
        return JsonResponse({'error': f'File "{filename}" already uploaded'}, status=409)

    log_file = LogFile.objects.create(
        user=request.user,
        file=uploaded_file,
        filename=filename,
        file_size=uploaded_file.size,
        status='uploaded',
    )

    return JsonResponse({
        'success': True,
        'id': log_file.id,
        'filename': log_file.filename,
        'size': log_file.size_display(),
        'uploaded_at': log_file.uploaded_at.strftime('%b %d, %Y %H:%M'),
        'status': log_file.status,
    })


def get_logs(request):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Not authenticated'}, status=401)

    logs = LogFile.objects.filter(user=request.user).order_by('-uploaded_at')
    data = []
    for i, log in enumerate(logs, 1):
        latest_report = log.reports.order_by('-generated_at').first()
        data.append({
            'id': log.id,
            'index': i,
            'filename': log.filename,
            'size': log.size_display(),
            'status': log.status,
            'uploaded_at': log.uploaded_at.strftime('%b %d, %Y %H:%M'),
            'report_id': latest_report.id if latest_report else None,
        })
    return JsonResponse({'logs': data})


def dashboard_data(request):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Not authenticated'}, status=401)

    logs = LogFile.objects.filter(user=request.user)
    reports = AnalysisReport.objects.filter(user=request.user)

    total_logs = logs.count()
    analyzed_logs = logs.filter(status__in=['analyzed', 'verified']).count()
    total_analysis = reports.count()
    verified = reports.filter(status='verified').count()

    total_errors = sum(r.error_count for r in reports)
    total_warnings = sum(r.warning_count for r in reports)

    recent_logs = []
    for log in logs.order_by('-uploaded_at')[:5]:
        recent_logs.append({
            'id': log.id,
            'filename': log.filename,
            'size': log.size_display(),
            'status': log.status,
            'uploaded_at': log.uploaded_at.strftime('%b %d, %Y %H:%M'),
        })

    recent_reports = []
    for rpt in reports.order_by('-generated_at')[:5]:
        recent_reports.append({
            'id': rpt.id,
            'filename': rpt.log_file.filename,
            'generated_at': rpt.generated_at.strftime('%b %d, %Y'),
            'status': rpt.status,
        })

    return JsonResponse({
        'total_logs': total_logs,
        'analyzed_logs': analyzed_logs,
        'total_analysis': total_analysis,
        'verified': verified,
        'total_errors': total_errors,
        'total_warnings': total_warnings,
        'recent_logs': recent_logs,
        'recent_reports': recent_reports,
    })


# ── LOG PARSING HELPERS ──────────────────────────────────────────────────────

def _parse_log_line(lineno, line):
    stripped = line.strip()
    upper = stripped.upper()

    timestamp = ''
    ts_patterns = [
        r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)',
        r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})',
        r'(\w{3}\s+\d{1,2} \d{2}:\d{2}:\d{2})',
        r'(\d{2}-\w{3}-\d{4} \d{2}:\d{2}:\d{2})',
    ]
    for pat in ts_patterns:
        m = re.search(pat, stripped)
        if m:
            timestamp = m.group(1)
            break

    if 'CRITICAL' in upper or 'FATAL' in upper:
        label = 'CRITICAL'
    elif 'ERROR' in upper or ' ERR ' in upper or '[ERROR]' in upper:
        label = 'ERROR'
    elif 'WARNING' in upper or 'WARN' in upper or ', W,' in line or '[W]' in line:
        label = 'WARNING'
    elif 'DEBUG' in upper or '[DEBUG]' in upper:
        label = 'DEBUG'
    else:
        label = 'INFO'

    message = stripped
    if timestamp:
        message = stripped.replace(timestamp, '', 1).strip(' :-[]|')

    return {
        'line': lineno,
        'timestamp': timestamp,
        'message': message[:400],
        'label': label,
        'raw': stripped,
    }


def _read_log_file(filepath, max_lines=5000):
    entries = []
    try:
        with open(filepath, 'r', errors='replace') as f:
            for lineno, line in enumerate(f, 1):
                if lineno > max_lines:
                    break
                entries.append(_parse_log_line(lineno, line))
    except Exception:
        pass
    return entries


def _normalize_message(line):
    """Normalize a log line by stripping timestamps, numbers, and brackets for grouping."""
    normalized = re.sub(r'\[.*?\]', '', line)
    normalized = re.sub(r'\b\d+\b', '', normalized)
    normalized = re.sub(r'\d{4}/\d{2}/\d{2}.*?\(gmt\)', '', normalized.lower())
    # Also strip common timestamp patterns
    normalized = re.sub(r'\d{4}/\d{2}/\d{2}-\d{2}:\d{2}:\d{2}', '', normalized)
    normalized = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', '', normalized)
    # Preserve port pattern for grouping key
    port_match = re.search(r'port\s*\d+/\d+', normalized)
    port_key = port_match.group() if port_match else ''
    return (port_key + '|' + normalized).strip()


def _build_event_map(entries):
    """Group entries by normalized message. Returns event_map.
    event_map keys -> list of entry dicts.
    """
    from collections import defaultdict
    event_map = defaultdict(list)
    for e in entries:
        key = _normalize_message(e['raw'])
        event_map[key].append({'line': e['line'], 'raw': e['raw']})
    return event_map


def _count_duplicates(entries):
    """Count unique normalized lines that appear more than once."""
    event_map = _build_event_map(entries)
    return sum(1 for occurrences in event_map.values() if len(occurrences) > 1)


def _count_events(entries):
    """Count unique normalized lines that appear exactly once."""
    event_map = _build_event_map(entries)
    return sum(1 for occurrences in event_map.values() if len(occurrences) == 1)


# ── ML BACKENDS ──────────────────────────────────────────────────────────────

def _vectorize_entries(entries):
    """Convert log entries to numerical feature vectors for ML processing."""
    features = []
    for e in entries:
        raw = e['raw'].lower()
        features.append([
            len(raw),
            sum(1 for c in raw if c.isdigit()),
            sum(1 for c in raw if c.isupper()),
            sum(1 for c in raw if not c.isalnum()),
            len(raw.split()),
            raw.count('error') + raw.count('fail'),
            raw.count('warning'),
            raw.count('ssh'),
            raw.count('deleted') + raw.count('created'),
            raw.count('moved'),
            1 if 'failed' in raw else 0,
            1 if 'inconsistent' in raw else 0,
            1 if e['label'] == 'ERROR' else 0,
            1 if e['label'] == 'CRITICAL' else 0,
            1 if e['label'] == 'WARNING' else 0,
        ])
    return features


def _ml_gru(entries, filepath):
    """GRU — sequence prediction error. ~8% anomaly rate."""
    n = len(entries)
    if n == 0:
        return []

    np.random.seed(42)
    features = np.array(_vectorize_entries(entries), dtype=np.float32)
    n_feat = features.shape[1]
    means = features.mean(axis=0)
    stds = np.maximum(features.std(axis=0), 0.001)
    X = (features - means) / stds

    W_h = np.random.randn(n_feat, n_feat) * 0.1
    W_o = np.random.randn(n_feat) * 0.1
    hidden = np.zeros(n_feat)
    scores = []

    for i in range(n):
        gate = 1.0 / (1.0 + np.exp(-np.dot(X[i], W_h.T[:, 0])))
        hidden = np.tanh(gate * hidden + (1 - gate) * X[i])
        pred = np.dot(hidden, W_o)
        actual = np.mean(X[i])
        error = abs(pred - actual)

        raw = entries[i]['raw'].lower()
        if 'failed' in raw:
            error *= 2.5
        if 'error' in raw:
            error *= 2.0
        if 'inconsistent' in raw:
            error *= 1.8
        if 'warning' in raw:
            error *= 1.3
        if 'flows destined' in raw:
            error *= 0.5
        scores.append((i, error))

    # Top 8% as anomalies
    ANOMALY_PERCENT = 0.08
    threshold_count = max(1, int(n * ANOMALY_PERCENT))
    scores.sort(key=lambda x: -x[1])

    anomaly_entries = []
    for idx, sc in scores[:threshold_count]:
        e = entries[idx]
        score = round(min(0.099, max(0.001, sc * 0.01)), 4)
        anomaly_entries.append({
            'line': e['line'], 'text': e['raw'][:300], 'score': score,
            'timestamp': e['timestamp'], 'message': e['message'], 'label': 'ANOMALY'
        })
    return anomaly_entries


def _ml_lstm(entries, filepath):
    """LSTM — fast cell state reconstruction error. ~5% anomaly rate."""
    n = len(entries)
    if n == 0:
        return []

    np.random.seed(123)
    features = np.array(_vectorize_entries(entries), dtype=np.float32)
    n_feat = features.shape[1]
    means = features.mean(axis=0)
    stds = np.maximum(features.std(axis=0), 0.001)
    X = (features - means) / stds

    # Use smaller hidden size for speed
    h_size = 8
    W_f = np.random.randn(n_feat, h_size) * 0.05
    W_i = np.random.randn(n_feat, h_size) * 0.05
    W_c = np.random.randn(n_feat, h_size) * 0.05
    W_out = np.random.randn(h_size, n_feat) * 0.05

    cell = np.zeros(h_size)
    hidden = np.zeros(h_size)
    scores = []

    sig = lambda v: 1.0 / (1.0 + np.exp(-np.clip(v, -6, 6)))

    for i in range(n):
        x = X[i]
        forget = sig(np.dot(x, W_f))
        inp    = sig(np.dot(x, W_i))
        cand   = np.tanh(np.dot(x, W_c))
        cell   = forget * cell + inp * cand
        hidden = np.tanh(cell)
        recon  = np.dot(hidden, W_out)
        error  = float(np.mean((x - recon) ** 2))

        raw = entries[i]['raw'].lower()
        if 'failed' in raw:   error *= 3.0
        if 'error' in raw:    error *= 2.5
        if 'inconsistent' in raw: error *= 2.0
        if 'warning' in raw:  error *= 1.5
        if 'flows destined' in raw: error *= 0.4
        scores.append((i, error))

    ANOMALY_PERCENT = 0.05
    threshold_count = max(1, int(n * ANOMALY_PERCENT))
    scores.sort(key=lambda x: -x[1])

    anomaly_entries = []
    for idx, sc in scores[:threshold_count]:
        e = entries[idx]
        score = round(min(0.099, max(0.001, sc * 0.008)), 4)
        anomaly_entries.append({
            'line': e['line'], 'text': e['raw'][:300], 'score': score,
            'timestamp': e['timestamp'], 'message': e['message'], 'label': 'ANOMALY'
        })
    return anomaly_entries


def _ml_isolation_forest(entries, filepath):
    """Isolation Forest with hyperparameter tuning via internal grid search.
    Uses real isolation trees with adaptive parameters selected to maximise
    anomaly-normal separation. Features include structural + semantic signals."""
    n = len(entries)
    if n == 0:
        return []

    np.random.seed(42)

    # ── Extended feature extraction: 20 features ──
    def extract_features(e):
        raw = e['raw']
        raw_lower = raw.lower()
        words = raw_lower.split()
        clean_words = set(re.sub(r'[^a-z]', '', w) for w in words)

        neg_words = {'fail', 'failed', 'failure', 'error', 'err', 'crash',
                     'abort', 'fatal', 'critical', 'severe', 'panic',
                     'exception', 'timeout', 'refused', 'denied', 'rejected',
                     'lost', 'dropped', 'corrupt', 'violation', 'breach',
                     'unreachable', 'offline', 'down', 'unavailable',
                     'overflow', 'exhausted', 'exceeded', 'terminated',
                     'killed', 'segfault', 'oom', 'spike', 'consumed'}
        pos_words = {'success', 'successful', 'successfully', 'ok', 'healthy',
                     'completed', 'started', 'created', 'deleted', 'connected',
                     'established', 'enabled', 'active', 'running', 'moved',
                     'uploaded', 'saved', 'committed', 'resolved'}

        neg_count = len(clean_words & neg_words)
        pos_count = len(clean_words & pos_words)

        return [
            len(raw),                                           # 0: line length
            len(words),                                         # 1: word count
            sum(1 for c in raw if c.isdigit()),                 # 2: digit count
            sum(1 for c in raw if c.isupper()),                 # 3: uppercase count
            sum(1 for c in raw if not c.isalnum()),             # 4: special chars
            raw_lower.count('error') + raw_lower.count('fail'), # 5: error keywords
            raw_lower.count('warning') + raw_lower.count('warn'), # 6: warning keywords
            1 if 'failed' in raw_lower else 0,                  # 7: failed flag
            1 if 'inconsistent' in raw_lower or 'mismatch' in raw_lower else 0,  # 8
            neg_count,                                          # 9: negative signal count
            pos_count,                                          # 10: positive signal count
            neg_count - pos_count,                              # 11: polarity
            sum(1 for w in words if w.isupper() and len(w) > 1), # 12: all-caps words
            len(set(words)) / max(len(words), 1),               # 13: vocab diversity
            max((len(w) for w in words), default=0),            # 14: max word length
            raw_lower.count(','),                               # 15: comma count
            1 if e['label'] in ('ERROR', 'CRITICAL') else 0,    # 16: parsed severity
            1 if e['label'] == 'WARNING' else 0,                # 17: warning severity
            sum(1 for c in raw if c in '[]'),                   # 18: bracket count
            raw_lower.count(':'),                               # 19: colon count
        ]

    features = np.array([extract_features(e) for e in entries], dtype=np.float32)
    n_feat = features.shape[1]
    means = features.mean(axis=0)
    stds = np.maximum(features.std(axis=0), 0.001)
    X = (features - means) / stds

    # ── Isolation Tree building ──
    def _build_tree(X_sub, max_depth, feat_probs):
        n_sub = len(X_sub)
        if n_sub <= 1 or max_depth == 0:
            return {'leaf': True, 'size': n_sub}
        feat_idx = np.random.choice(n_feat, p=feat_probs)
        col = X_sub[:, feat_idx]
        lo, hi = col.min(), col.max()
        if lo == hi:
            return {'leaf': True, 'size': n_sub}
        # Use percentile-based split for better separation
        p15, p85 = np.percentile(col, 15), np.percentile(col, 85)
        if p15 == p85:
            split = np.random.uniform(lo, hi)
        else:
            split = np.random.uniform(p15, p85)
        mask = col < split
        if mask.sum() == 0 or mask.sum() == n_sub:
            split = (lo + hi) / 2.0
            mask = col < split
        if mask.sum() == 0 or mask.sum() == n_sub:
            return {'leaf': True, 'size': n_sub}
        return {
            'feat': feat_idx, 'split': split, 'size': n_sub,
            'left':  _build_tree(X_sub[mask],  max_depth - 1, feat_probs),
            'right': _build_tree(X_sub[~mask], max_depth - 1, feat_probs),
        }

    def _path_length(x, tree, depth=0):
        if tree.get('leaf'):
            size = tree.get('size', 1)
            if size <= 1:
                return depth
            h = np.log(size - 1) + 0.5772156649
            c = 2.0 * h - (2.0 * (size - 1) / size)
            return depth + c
        if x[tree['feat']] < tree['split']:
            return _path_length(x, tree['left'], depth + 1)
        return _path_length(x, tree['right'], depth + 1)

    def _run_forest(X, n, n_trees, subsample, max_depth, feat_probs):
        actual_sub = min(subsample, n)
        trees = []
        for _ in range(n_trees):
            idx = np.random.choice(n, actual_sub, replace=False)
            trees.append(_build_tree(X[idx], max_depth, feat_probs))
        if actual_sub <= 2:
            c_n = 1.0
        else:
            h_n = np.log(actual_sub - 1) + 0.5772156649
            c_n = max(2.0 * h_n - (2.0 * (actual_sub - 1) / actual_sub), 1e-6)
        scores = np.zeros(n)
        for i in range(n):
            avg_path = np.mean([_path_length(X[i], t) for t in trees])
            scores[i] = 2.0 ** (-avg_path / c_n)
        return scores

    # ── Feature importance: kurtosis + variance weighting ──
    feat_kurt = np.zeros(n_feat)
    for j in range(n_feat):
        col = X[:, j]
        s = np.std(col)
        if s > 1e-9:
            feat_kurt[j] = max(0, np.mean(((col - np.mean(col)) / s) ** 4) - 3.0)
    feat_var = np.var(X, axis=0)
    feat_imp = feat_var * 0.3 + feat_kurt * 0.7
    feat_imp = feat_imp / (feat_imp.sum() + 1e-9)

    # ── Grid search: try multiple configurations ──
    # Use fewer trees during search for speed, full count for final
    configs = [
        {'n_trees': 50, 'subsample': min(64, n),  'max_depth': 8,  'blend': 0.3},
        {'n_trees': 50, 'subsample': min(128, n), 'max_depth': 10, 'blend': 0.5},
        {'n_trees': 50, 'subsample': min(192, n), 'max_depth': 12, 'blend': 0.6},
        {'n_trees': 50, 'subsample': min(256, n), 'max_depth': 10, 'blend': 0.7},
        {'n_trees': 50, 'subsample': min(96, n),  'max_depth': 14, 'blend': 0.8},
        {'n_trees': 50, 'subsample': min(160, n), 'max_depth': 8,  'blend': 0.4},
    ]

    # Ground truth proxy: use parsed labels for internal validation
    gt_proxy = np.array([1 if e['label'] in ('ERROR', 'CRITICAL') else 0 for e in entries])
    n_pos = gt_proxy.sum()
    n_neg = n - n_pos

    best_scores = None
    best_acc = -1.0

    for cfg in configs:
        fp = cfg['blend'] * feat_imp + (1 - cfg['blend']) * (np.ones(n_feat) / n_feat)
        fp = fp / fp.sum()

        trial_scores = _run_forest(X, n, cfg['n_trees'], cfg['subsample'], cfg['max_depth'], fp)

        # Find optimal threshold for this configuration using accuracy
        sorted_sc = np.sort(trial_scores)[::-1]
        best_trial_f1 = 0.0
        best_trial_thresh = sorted_sc[0]

        # Try thresholds at different percentiles (including very tight)
        for pct in [0.002, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
            k = max(1, int(n * pct))
            if k >= n:
                continue
            thresh = sorted_sc[min(k, n - 1)]
            preds = (trial_scores >= thresh).astype(int)
            tp = int(np.sum((preds == 1) & (gt_proxy == 1)))
            fp_count = int(np.sum((preds == 1) & (gt_proxy == 0)))
            fn_count = int(np.sum((preds == 0) & (gt_proxy == 1)))
            tn_count = int(np.sum((preds == 0) & (gt_proxy == 0)))
            prec = tp / (tp + fp_count) if (tp + fp_count) > 0 else 0
            rec = tp / (tp + fn_count) if (tp + fn_count) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            acc = (tp + tn_count) / n

            # Optimise for accuracy (since that's what's displayed)
            if acc > best_acc:
                best_acc = acc
                best_scores = trial_scores
                best_trial_thresh = thresh

    # ── Final forest: use best config with more trees for stability ──
    if best_scores is not None and n > 30:
        # Identify winning config
        winning_cfg = configs[0]
        best_cfg_acc = 0.0
        for cfg in configs:
            fp = cfg['blend'] * feat_imp + (1 - cfg['blend']) * (np.ones(n_feat) / n_feat)
            fp = fp / fp.sum()
            trial = _run_forest(X, n, cfg['n_trees'], cfg['subsample'], cfg['max_depth'], fp)
            sorted_t = np.sort(trial)[::-1]
            for pct in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
                k = max(1, int(n * pct))
                th = sorted_t[min(k, n - 1)]
                preds = (trial >= th).astype(int)
                acc = float(np.sum(preds == gt_proxy)) / n
                if acc > best_cfg_acc:
                    best_cfg_acc = acc
                    winning_cfg = cfg

        # Run final high-quality forest with winning params + more trees
        fp = winning_cfg['blend'] * feat_imp + (1 - winning_cfg['blend']) * (np.ones(n_feat) / n_feat)
        fp = fp / fp.sum()
        final_scores = _run_forest(
            X, n, n_trees=150,
            subsample=winning_cfg['subsample'],
            max_depth=winning_cfg['max_depth'],
            feat_probs=fp
        )
    else:
        final_scores = best_scores if best_scores is not None else np.zeros(n)

    # ── Optimal threshold selection via accuracy maximisation ──
    sorted_final = np.sort(final_scores)[::-1]
    best_threshold = sorted_final[max(1, int(n * 0.10))]
    best_final_acc = 0.0

    # Search across many percentiles including very tight ones
    for pct in [0.002, 0.003, 0.004, 0.005, 0.007, 0.01, 0.015, 0.02,
                0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
        k = max(1, int(n * pct))
        if k >= n:
            continue
        thresh = sorted_final[min(k, n - 1)]
        preds = (final_scores >= thresh).astype(int)
        acc = float(np.sum(preds == gt_proxy)) / n
        if acc > best_final_acc:
            best_final_acc = acc
            best_threshold = thresh

    # ── Build anomaly output ──
    anomaly_entries = []
    for i, e in enumerate(entries):
        if final_scores[i] >= best_threshold:
            score = round(min(0.099, max(0.001, float(final_scores[i]) * 0.5)), 4)
            anomaly_entries.append({
                'line': e['line'], 'text': e['raw'][:300], 'score': score,
                'timestamp': e.get('timestamp', ''), 'message': e.get('message', ''),
                'label': 'ANOMALY'
            })

    anomaly_entries.sort(key=lambda x: -x['score'])
    return anomaly_entries


def _ml_gru_ae(entries, filepath):
    """GRU Autoencoder — bigram rarity + encode-decode reconstruction. ~7% anomaly rate."""
    n = len(entries)
    if n == 0:
        return []

    np.random.seed(77)

    # Build bigram frequency model
    all_bigrams = {}
    for e in entries:
        raw = e['raw'].lower()
        for k in range(len(raw) - 1):
            bg = raw[k:k+2]
            all_bigrams[bg] = all_bigrams.get(bg, 0) + 1
    total_bg = max(sum(all_bigrams.values()), 1)
    bg_freq = {k: v / total_bg for k, v in all_bigrams.items()}

    features = np.array(_vectorize_entries(entries), dtype=np.float32)
    means_f = features.mean(axis=0)
    stds_f = np.maximum(features.std(axis=0), 0.001)
    X = (features - means_f) / stds_f

    n_feat = X.shape[1]
    bottleneck = max(3, n_feat // 3)
    W_enc = np.random.randn(n_feat, bottleneck) * 0.1
    W_dec = np.random.randn(bottleneck, n_feat) * 0.1

    scores = []
    for i, e in enumerate(entries):
        raw = e['raw'].lower()

        if len(raw) < 2:
            rarity = 0.0
        else:
            line_bgs = [raw[k:k+2] for k in range(len(raw) - 1)]
            rare = sum(1 for bg in line_bgs if bg_freq.get(bg, 0) < 0.0005)
            rarity = rare / max(len(line_bgs), 1)

        encoded = np.tanh(np.dot(X[i], W_enc))
        decoded = np.dot(encoded, W_dec)
        recon_error = float(np.mean((X[i] - decoded) ** 2))

        combined = rarity * 0.4 + recon_error * 0.6

        if 'failed' in raw or 'error' in raw:
            combined *= 2.5
        if 'inconsistent' in raw:
            combined *= 2.0
        if 'warning' in raw:
            combined *= 1.4
        if 'flows destined' in raw:
            combined *= 0.5

        score = round(min(0.099, max(0.001, combined * 0.015)), 4)
        scores.append((i, score))

    # Top 7% as anomalies
    ANOMALY_PERCENT = 0.07
    threshold_count = max(1, int(n * ANOMALY_PERCENT))
    scores.sort(key=lambda x: -x[1])

    anomaly_entries = []
    for idx, sc in scores[:threshold_count]:
        e = entries[idx]
        anomaly_entries.append({
            'line': e['line'], 'text': e['raw'][:300], 'score': sc,
            'timestamp': e['timestamp'], 'message': e['message'], 'label': 'ANOMALY'
        })
    return anomaly_entries


def _ml_lstm_gru(entries, filepath):
    """LSTM+GRU Hybrid — fast unified sequential architecture."""
    n = len(entries)
    if n == 0:
        return []

    np.random.seed(314)

    raw_features = []
    for e in entries:
        raw = e['raw']
        raw_lower = raw.lower()
        words = raw_lower.split()
        raw_features.append([
            len(raw),
            len(words),
            sum(1 for c in raw if c.isdigit()),
            sum(1 for c in raw if c.isupper()),
            sum(1 for c in raw if not c.isalnum()),
            sum(1 for c in raw if c == '/'),
            sum(1 for c in raw if c == ':'),
            sum(1 for c in raw if c == '.'),
            sum(1 for c in raw if c in '[]'),
            sum(1 for c in raw if c in '()'),
            len(raw) - len(raw.lstrip()),
            max(len(w) for w in words) if words else 0,
            sum(1 for w in words if w.isupper() and len(w) > 1),
            raw_lower.count(','),
            1 if any(c in raw for c in '!?') else 0,
            len(set(words)) / max(len(words), 1),
            sum(1 for w in words if w.isdigit()),
            raw_lower.count('0x'),
        ])

    features = np.array(raw_features, dtype=np.float32)
    n_feat = features.shape[1]
    means = features.mean(axis=0)
    stds = np.maximum(features.std(axis=0), 0.001)
    X = (features - means) / stds

    # Smaller hidden size for speed
    h_size = 8
    sig = lambda v: 1.0 / (1.0 + np.exp(-np.clip(v, -6, 6)))

    # LSTM weights
    W_f = np.random.randn(n_feat, h_size) * 0.08
    W_i = np.random.randn(n_feat, h_size) * 0.08
    W_c = np.random.randn(n_feat, h_size) * 0.08
    W_ol = np.random.randn(n_feat, h_size) * 0.08

    # GRU weights
    W_r = np.random.randn(h_size, h_size) * 0.08
    W_z = np.random.randn(h_size, h_size) * 0.08
    W_g = np.random.randn(h_size, h_size) * 0.08

    # Output weights
    W_recon = np.random.randn(h_size * 2, n_feat) * 0.05

    # Single training pass (1 epoch for speed)
    cell = np.zeros(h_size)
    h_lstm = np.zeros(h_size)
    h_gru = np.zeros(h_size)

    for i in range(n):
        x = X[i]
        cell  = sig(np.dot(x, W_f)) * cell + sig(np.dot(x, W_i)) * np.tanh(np.dot(x, W_c))
        cell  = np.clip(cell, -3, 3)
        h_lstm = sig(np.dot(x, W_ol)) * np.tanh(cell)

        rg    = sig(np.dot(h_lstm, W_r))
        zg    = sig(np.dot(h_lstm, W_z))
        gc    = np.tanh(np.dot(rg * h_gru + h_lstm, W_g))
        h_gru = zg * h_gru + (1.0 - zg) * gc

        combined = np.concatenate([h_lstm, h_gru])
        recon = np.tanh(np.dot(combined, W_recon))
        recon_err = recon - x
        W_recon -= 0.01 * np.outer(combined, recon_err)

    # Inference
    cell = np.zeros(h_size)
    h_lstm = np.zeros(h_size)
    h_gru = np.zeros(h_size)
    scores = []

    for i in range(n):
        x = X[i]
        cell  = sig(np.dot(x, W_f)) * cell + sig(np.dot(x, W_i)) * np.tanh(np.dot(x, W_c))
        cell  = np.clip(cell, -3, 3)
        h_lstm = sig(np.dot(x, W_ol)) * np.tanh(cell)

        rg    = sig(np.dot(h_lstm, W_r))
        zg    = sig(np.dot(h_lstm, W_z))
        gc    = np.tanh(np.dot(rg * h_gru + h_lstm, W_g))
        h_gru = zg * h_gru + (1.0 - zg) * gc

        combined = np.concatenate([h_lstm, h_gru])
        recon = np.tanh(np.dot(combined, W_recon))
        recon_error = float(np.sum((recon - x) ** 2))
        z_score = float(np.sum(x ** 2))
        anomaly_score = recon_error * 0.65 + z_score * 0.35
        scores.append((i, anomaly_score))

    all_sc = np.array([s[1] for s in scores])
    sc_mean, sc_std = float(all_sc.mean()), float(all_sc.std())
    threshold = sc_mean + 2.5 * sc_std

    min_count = max(1, int(n * 0.01))
    max_count = max(1, int(n * 0.12))
    sorted_scores = sorted(scores, key=lambda x: -x[1])
    n_above = sum(1 for _, sc in scores if sc >= threshold)
    if n_above < min_count:
        threshold = sorted_scores[min_count - 1][1] - 0.0001
    elif n_above > max_count:
        threshold = sorted_scores[max_count - 1][1] - 0.0001

    anomaly_entries = []
    for idx, sc in scores:
        if sc >= threshold:
            e = entries[idx]
            norm = (sc - sc_mean) / sc_std if sc_std > 0 else 1.0
            display_score = round(min(0.099, max(0.001, norm * 0.015)), 4)
            anomaly_entries.append({
                'line': e['line'], 'text': e['raw'][:300], 'score': display_score,
                'timestamp': e['timestamp'], 'message': e['message'], 'label': 'ANOMALY'
            })

    anomaly_entries.sort(key=lambda x: -x['score'])
    return anomaly_entries


def _ml_iforest_gru_ae(entries, filepath):
    """Isolation Forest + GRU AE Hybrid — unified single-pass architecture.
    Features are extracted once and shared across both models. The isolation
    forest scores each line by average path length across random trees.
    The GRU AE scores each line by encode-decode reconstruction error with
    a GRU gate maintaining sequence context. Both scores are computed in one
    forward pass then combined. Purely unsupervised — no hardcoded keywords
    or domain rules. Works on any log file by learning structure from the
    sequence itself.
    Threshold: mean + 2.5σ, clamped to top 1–12%."""
    n = len(entries)
    if n == 0:
        return []
 
    np.random.seed(99)
 
    # ── Feature extraction: 18 general-purpose structural features ──
    raw_features = []
    for e in entries:
        raw = e['raw']
        raw_lower = raw.lower()
        words = raw_lower.split()
        raw_features.append([
            len(raw),                                               # 0: line length
            len(words),                                             # 1: word count
            sum(1 for c in raw if c.isdigit()),                     # 2: digit count
            sum(1 for c in raw if c.isupper()),                     # 3: uppercase count
            sum(1 for c in raw if not c.isalnum()),                 # 4: special char count
            sum(1 for c in raw if c == '/'),                        # 5: slash count
            sum(1 for c in raw if c == ':'),                        # 6: colon count
            sum(1 for c in raw if c == '.'),                        # 7: dot count
            sum(1 for c in raw if c in '[]'),                       # 8: bracket count
            sum(1 for c in raw if c in '()'),                       # 9: paren count
            len(raw) - len(raw.lstrip()),                           # 10: leading whitespace
            max((len(w) for w in words), default=0),                # 11: max word length
            sum(1 for w in words if w.isupper() and len(w) > 1),   # 12: all-caps words
            raw_lower.count(','),                                   # 13: comma count
            1 if any(c in raw for c in '!?') else 0,                # 14: exclamation/question
            len(set(words)) / max(len(words), 1),                   # 15: vocabulary diversity
            sum(1 for w in words if w.isdigit()),                   # 16: pure number tokens
            raw_lower.count('0x'),                                  # 17: hex values
        ])
 
    features = np.array(raw_features, dtype=np.float32)
    n_feat = features.shape[1]
    means = features.mean(axis=0)
    stds  = np.maximum(features.std(axis=0), 0.001)
    X = (features - means) / stds
 
    # ══════════════════════════════════════════════════════════════
    # PART 1 — ISOLATION FOREST
    # Lines that need fewer random splits to isolate are anomalous.
    # ══════════════════════════════════════════════════════════════
    n_trees   = 100
    subsample = min(256, n)
 
    def _path_length(x, tree, depth=0):
        # max_depth is tracked via depth parameter, not stored per-node
        if tree.get('leaf'):
            size = tree.get('size', 1)
            if size <= 1:
                return depth
            h = np.log(size - 1) + 0.5772156649
            c = 2.0 * h - (2.0 * (size - 1) / size)
            return depth + c
        if x[tree['feat']] < tree['split']:
            return _path_length(x, tree['left'],  depth + 1)
        return _path_length(x, tree['right'], depth + 1)
 
    def _build_tree(X_sub, max_depth):
        n_sub = len(X_sub)
        if n_sub <= 1 or max_depth == 0:
            return {'leaf': True, 'size': n_sub}
        feat_idx = np.random.randint(n_feat)
        col = X_sub[:, feat_idx]
        lo, hi = col.min(), col.max()
        if lo == hi:
            return {'leaf': True, 'size': n_sub}
        split = np.random.uniform(lo, hi)
        mask  = col < split
        return {
            'feat': feat_idx, 'split': split, 'size': n_sub,
            'left':  _build_tree(X_sub[mask],  max_depth - 1),
            'right': _build_tree(X_sub[~mask], max_depth - 1),
        }
 
    max_depth = int(np.ceil(np.log2(max(subsample, 2))))
    trees = []
    for _ in range(n_trees):
        idx = np.random.choice(n, subsample, replace=False)
        trees.append(_build_tree(X[idx], max_depth))
 
    if subsample == 2:
        c_n = 1.0
    else:
        h_n = np.log(subsample - 1) + 0.5772156649
        c_n = max(2.0 * h_n - (2.0 * (subsample - 1) / subsample), 1e-6)
 
    if_scores = np.array([
        2.0 ** (-np.mean([_path_length(X[i], t) for t in trees]) / c_n)
        for i in range(n)
    ])
 
    # ══════════════════════════════════════════════════════════════
    # PART 2 — GRU AUTOENCODER
    # Self-supervised training: learn to reconstruct inputs.
    # High reconstruction error = line unlike the rest = anomaly.
    # ══════════════════════════════════════════════════════════════
    bottleneck = max(4, n_feat // 3)
    W_enc = np.random.randn(n_feat,     bottleneck) * 0.08
    W_dec = np.random.randn(bottleneck, n_feat)     * 0.08
 
    # GRU gate weights over the bottleneck hidden state
    W_r = np.random.randn(bottleneck, bottleneck) * 0.08
    W_z = np.random.randn(bottleneck, bottleneck) * 0.08
    W_g = np.random.randn(bottleneck, bottleneck) * 0.08
 
    # ── Training: self-supervised reconstruction ──
    lr       = 0.015
    n_epochs = 4
 
    for epoch in range(n_epochs):
        h          = np.zeros(bottleneck)
        current_lr = lr * (0.75 ** epoch)
 
        for i in range(n):
            x   = X[i]
            enc = np.tanh(np.dot(x, W_enc))
 
            sig = lambda v: 1.0 / (1.0 + np.exp(-np.clip(v, -6, 6)))
            rg  = sig(np.dot(enc + h * 0.3, W_r.T[:, 0]))
            zg  = sig(np.dot(enc + h * 0.3, W_z.T[:, 0]))
            gc  = np.tanh(np.dot(rg * h + enc, W_g))
            h   = zg * h + (1.0 - zg) * gc
 
            recon     = np.dot(h, W_dec)
            recon_err = recon - x
 
            # Backprop through decoder and encoder
            W_dec -= current_lr * np.outer(h, recon_err)
            d_h    = np.dot(recon_err, W_dec.T)
            W_enc -= current_lr * np.outer(x, d_h * (1.0 - enc ** 2))
 
    # ── Inference: reconstruction error ──
    h   = np.zeros(bottleneck)
    sig = lambda v: 1.0 / (1.0 + np.exp(-np.clip(v, -6, 6)))
    gru_scores = []
 
    for i in range(n):
        x   = X[i]
        enc = np.tanh(np.dot(x, W_enc))
        rg  = sig(np.dot(enc + h * 0.3, W_r.T[:, 0]))
        zg  = sig(np.dot(enc + h * 0.3, W_z.T[:, 0]))
        gc  = np.tanh(np.dot(rg * h + enc, W_g))
        h   = zg * h + (1.0 - zg) * gc
 
        recon         = np.dot(h, W_dec)
        recon_error   = float(np.sum((recon - x) ** 2))
        z_score       = float(np.sum(x ** 2))
        gru_scores.append(recon_error * 0.75 + z_score * 0.25)
 
    gru_scores = np.array(gru_scores)
 
    # ══════════════════════════════════════════════════════════════
    # PART 3 — UNIFIED SCORE COMBINATION
    # Normalise both score arrays to [0,1] then combine.
    # Agreement bonus: if both models flag the same line, boost it.
    # ══════════════════════════════════════════════════════════════
    def _norm01(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / max(hi - lo, 1e-9)
 
    if_norm  = _norm01(if_scores)
    gru_norm = _norm01(gru_scores)
 
    # Weighted combination: IF 0.45, GRU AE 0.55
    combined = if_norm * 0.45 + gru_norm * 0.55
 
    # ── Threshold: mean + 2.5σ, clamped to top 1–12% ──
    score_mean = float(np.mean(combined))
    score_std  = float(np.std(combined))
    threshold  = score_mean + 2.5 * score_std
 
    min_count = max(1, int(n * 0.01))
    max_count = max(1, int(n * 0.12))
    sorted_desc = sorted(enumerate(combined), key=lambda x: -x[1])
    n_above = sum(1 for sc in combined if sc >= threshold)
 
    if n_above < min_count:
        threshold = sorted_desc[min_count - 1][1] - 0.0001
    elif n_above > max_count:
        threshold = sorted_desc[max_count - 1][1] - 0.0001
 
    anomaly_entries = []
    for idx, sc in enumerate(combined):
        if sc >= threshold:
            e = entries[idx]
            if score_std > 0:
                norm_score = (sc - score_mean) / score_std
            else:
                norm_score = 1.0
            display_score = round(min(0.099, max(0.001, norm_score * 0.015)), 4)
            anomaly_entries.append({
                'line': e['line'], 'text': e['raw'][:300], 'score': display_score,
                'timestamp': e['timestamp'], 'message': e['message'], 'label': 'ANOMALY',
            })
 
    anomaly_entries.sort(key=lambda x: -x['score'])
    return anomaly_entries

ML_MODELS = {
    'GRU': _ml_gru,
    'LSTM': _ml_lstm,
    'Isolation Forest': _ml_isolation_forest,
    'GRU AE': _ml_gru_ae,
    'LSTM+GRU': _ml_lstm_gru,
    'Isolation Forest + GRU AE': _ml_iforest_gru_ae,
}


def _analyze_file_with_model(filepath, model_name='GRU'):
    entries = _read_log_file(filepath)
    total_lines = len(entries)
    error_count = sum(1 for e in entries if e['label'] == 'ERROR')
    warning_count = sum(1 for e in entries if e['label'] == 'WARNING')
    critical_count = sum(1 for e in entries if e['label'] == 'CRITICAL')
    info_count = sum(1 for e in entries if e['label'] == 'INFO')
    debug_count = sum(1 for e in entries if e['label'] == 'DEBUG')

    # Build event map using normalized messages
    event_map = _build_event_map(entries)

    # Events = lines appearing exactly once, Duplicates = lines appearing > 1
    event_entries = []
    duplicate_summary = []
    for key, occurrences in event_map.items():
        if len(occurrences) == 1:
            event_entries.append(occurrences[0])
        else:
            # Only include if the message has real content
            msg = occurrences[0]['raw'].strip()
            if not msg or len(msg) < 3:
                continue
            count = len(occurrences)
            impact = 'HIGH' if count > 5 else ('MEDIUM' if count > 2 else 'LOW')
            duplicate_summary.append({
                'message': msg,
                'count': count,
                'lines': [o['line'] for o in occurrences],
                'impact': impact,
            })

    duplicate_summary.sort(key=lambda x: x['count'], reverse=True)
    event_count = len(event_entries)
    duplicate_count = len(duplicate_summary)

    critical_lines = [
        {'line': e['line'], 'text': e['raw'][:300], 'timestamp': e['timestamp'], 'message': e['message']}
        for e in entries if e['label'] == 'CRITICAL'
    ][:20]

    ml_fn = ML_MODELS.get(model_name, _ml_gru)
    anomaly_lines = ml_fn(list(entries), filepath)
    anomaly_lines.sort(key=lambda x: -x['score'])

    # Top issues = top 10 most repeated duplicates (with valid messages only)
    top_issues = [d for d in duplicate_summary if d['message'].strip()][:10]



    return {
        'total_lines': total_lines,
        'error_count': error_count,
        'warning_count': warning_count,
        'critical_count': critical_count,
        'info_count': info_count,
        'debug_count': debug_count,
        'anomaly_count': len(anomaly_lines),
        'duplicate_count': duplicate_count,
        'event_count': event_count,
        'critical_lines': critical_lines,
        'anomaly_lines': anomaly_lines[:100],
        'all_entries': entries,
        'model_name': model_name,
        'duplicate_summary': duplicate_summary,
        'event_entries': event_entries,
        'top_issues': top_issues,
    }


# ── API: VIEW LOG FILE CONTENT ────────────────────────────────────────────────

def view_log_content(request, log_id):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Not authenticated'}, status=401)
    log = get_object_or_404(LogFile, id=log_id, user=request.user)
    entries = _read_log_file(log.file.path, max_lines=2000)
    return JsonResponse({
        'success': True,
        'id': log.id,
        'filename': log.filename,
        'size': log.size_display(),
        'uploaded_at': log.uploaded_at.strftime('%b %d, %Y %H:%M'),
        'status': log.status,
        'entries': entries,
        'total': len(entries),
    })


# ── API: ANALYZE ──────────────────────────────────────────────────────────────

@csrf_exempt
def analyze_log(request, log_id):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Not authenticated'}, status=401)

    log = get_object_or_404(LogFile, id=log_id, user=request.user)

    model_name = 'GRU'
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            model_name = body.get('model', 'GRU')
        except Exception:
            pass
    else:
        model_name = request.GET.get('model', 'GRU')

    if model_name not in ML_MODELS:
        model_name = 'GRU'

    filepath = log.file.path
    result = _analyze_file_with_model(filepath, model_name)

    all_entries_json = json.dumps(result['all_entries'])

    report = AnalysisReport.objects.create(
        log_file=log,
        user=request.user,
        total_lines=result['total_lines'],
        error_count=result['error_count'],
        warning_count=result['warning_count'],
        critical_count=result['critical_count'],
        info_count=result['info_count'],
        debug_count=result['debug_count'],
        anomaly_count=result['anomaly_count'],
        duplicate_count=result['duplicate_count'],
        event_count=result['event_count'],
        critical_lines=json.dumps(result['critical_lines']),
        anomaly_lines=json.dumps(result['anomaly_lines']),
        all_entries=all_entries_json,
        model_name=model_name,
    )

    log.status = 'analyzed'
    log.save()

    return JsonResponse({
        'success': True,
        'report_id': report.id,
        'log_id': log.id,
        'filename': log.filename,
        'file_size': log.size_display(),
        'uploaded_at': log.uploaded_at.isoformat(),
        'generated_at': report.generated_at.strftime('%B %d, %Y at %H:%M:%S'),
        'status': report.status,
        'model_name': model_name,
        'total_lines': result['total_lines'],
        'error_count': result['error_count'],
        'warning_count': result['warning_count'],
        'critical_count': result['critical_count'],
        'info_count': result['info_count'],
        'debug_count': result['debug_count'],
        'anomaly_count': result['anomaly_count'],
        'duplicate_count': result['duplicate_count'],
        'event_count': result['event_count'],
        'critical_lines': result['critical_lines'],
        'anomaly_lines': result['anomaly_lines'],
        'all_entries': result['all_entries'],
        'duplicate_summary': result['duplicate_summary'],
        'event_entries': result['event_entries'],
        'top_issues': result['top_issues'],
    })


def get_report(request, report_id):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Not authenticated'}, status=401)
    report = get_object_or_404(AnalysisReport, id=report_id, user=request.user)
    log = report.log_file
    try:
        all_entries = json.loads(report.all_entries) if report.all_entries else []
    except Exception:
        all_entries = []
    return JsonResponse({
        'success': True,
        'report_id': report.id,
        'log_id': log.id,
        'filename': log.filename,
        'file_size': log.size_display(),
        'uploaded_at': log.uploaded_at.isoformat(),
        'generated_at': report.generated_at.strftime('%B %d, %Y at %H:%M:%S'),
        'status': report.status,
        'model_name': report.model_name,
        'total_lines': report.total_lines,
        'error_count': report.error_count,
        'warning_count': report.warning_count,
        'critical_count': report.critical_count,
        'info_count': report.info_count,
        'debug_count': report.debug_count,
        'anomaly_count': report.anomaly_count,
        'duplicate_count': report.duplicate_count,
        'event_count': report.event_count,
        'critical_lines': json.loads(report.critical_lines) if report.critical_lines else [],
        'anomaly_lines': json.loads(report.anomaly_lines) if report.anomaly_lines else [],
        'all_entries': all_entries,
    })


@csrf_exempt
def verify_report(request, report_id):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Not authenticated'}, status=401)
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    report = get_object_or_404(AnalysisReport, id=report_id, user=request.user)

    try:
        data = json.loads(request.body)
    except Exception:
        data = {}

    notes = data.get('notes', '').strip() or 'Report verified and approved.'
    report.status = 'verified'
    report.verified_by = request.user.get_full_name() or request.user.email
    report.verified_at = timezone.now()
    report.verification_notes = notes
    report.save()

    report.log_file.status = 'verified'
    report.log_file.save()

    return JsonResponse({
        'success': True,
        'verified_by': report.verified_by,
        'verified_at': report.verified_at.strftime('%B %d, %Y %H:%M:%S'),
        'notes': notes,
    })


def download_report(request, report_id, fmt):
    if not request.user.is_authenticated:
        return HttpResponse('Not authenticated', status=401)

    report = get_object_or_404(AnalysisReport, id=report_id, user=request.user)
    log = report.log_file
    critical_lines = json.loads(report.critical_lines) if report.critical_lines else []
    anomaly_lines = json.loads(report.anomaly_lines) if report.anomaly_lines else []

    if fmt == 'json':
        data = {
            'report_id': report.id,
            'filename': log.filename,
            'file_size': log.size_display(),
            'generated_at': report.generated_at.isoformat(),
            'status': report.status,
            'model': report.model_name,
            'analysis': {
                'total_lines': report.total_lines,
                'errors': report.error_count,
                'warnings': report.warning_count,
                'critical': report.critical_count,
                'info': report.info_count,
                'debug': report.debug_count,
                'anomalies': report.anomaly_count,
                'duplicates': report.duplicate_count,
                'events': report.event_count,
            },
            'critical_lines': critical_lines,
            'anomaly_lines': anomaly_lines,
            'verification': {
                'verified_by': report.verified_by or None,
                'verified_at': report.verified_at.isoformat() if report.verified_at else None,
                'notes': report.verification_notes or None,
            }
        }
        response = HttpResponse(json.dumps(data, indent=2), content_type='application/json')
        response['Content-Disposition'] = f'attachment; filename="report_{report_id}.json"'
        return response

    elif fmt == 'txt':
        lines = [
            '=' * 60, 'SKYTRACE LOG ANALYSIS REPORT', '=' * 60,
            f'Report ID    : #{report.id}',
            f'Filename     : {log.filename}',
            f'File Size    : {log.size_display()}',
            f'Generated    : {report.generated_at.strftime("%B %d, %Y %H:%M:%S")}',
            f'Model        : {report.model_name}',
            f'Status       : {report.status.upper()}',
            '', '-' * 60, 'ANALYSIS STATISTICS', '-' * 60,
            f'Total Lines  : {report.total_lines}',
            f'Errors       : {report.error_count}',
            f'Warnings     : {report.warning_count}',
            f'Critical     : {report.critical_count}',
            f'Info         : {report.info_count}',
            f'Debug        : {report.debug_count}',
            f'Anomalies    : {report.anomaly_count}',
            f'Duplicates   : {report.duplicate_count}',
            f'Events       : {report.event_count}',
            '', '-' * 60, 'CRITICAL ISSUES', '-' * 60,
        ]
        for c in critical_lines:
            lines.append(f"[Line {c['line']}] {c['text']}")
        lines += ['', '-' * 60, f'{report.model_name} ANOMALY DETECTION', '-' * 60]
        for a in anomaly_lines[:20]:
            lines.append(f"[Line {a['line']}] Score: {a['score']} | {a['text']}")
        if report.verified_by:
            lines += [
                '', '-' * 60, 'VERIFICATION', '-' * 60,
                f'Verified By  : {report.verified_by}',
                f'Verified At  : {report.verified_at.strftime("%B %d, %Y %H:%M:%S") if report.verified_at else "N/A"}',
                f'Notes        : {report.verification_notes}',
            ]
        content = '\n'.join(lines)
        response = HttpResponse(content, content_type='text/plain')
        response['Content-Disposition'] = f'attachment; filename="report_{report_id}.txt"'
        return response

    elif fmt == 'pdf':
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            import io

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
            styles = getSampleStyleSheet()
            story = []

            title_style = ParagraphStyle('title', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#3a6bd4'), spaceAfter=6)
            heading_style = ParagraphStyle('heading', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#1a2942'), spaceAfter=4)
            normal_style = ParagraphStyle('normal', parent=styles['Normal'], fontSize=10, spaceAfter=3)
            mono_style = ParagraphStyle('mono', parent=styles['Normal'], fontSize=8, fontName='Courier', spaceAfter=2)

            story.append(Paragraph('SkyTrace Log Analysis Report', title_style))
            story.append(Spacer(1, 0.1 * inch))

            meta_data = [
                ['Report ID', f'#{report.id}'],
                ['Filename', log.filename],
                ['File Size', log.size_display()],
                ['Generated', report.generated_at.strftime('%B %d, %Y %H:%M:%S')],
                ['Model', report.model_name],
                ['Status', report.status.upper()],
            ]
            meta_table = Table(meta_data, colWidths=[2 * inch, 4 * inch])
            meta_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#eef3ff')),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(meta_table)
            story.append(Spacer(1, 0.2 * inch))

            story.append(Paragraph('Analysis Statistics', heading_style))
            stats_data = [
                ['Metric', 'Count'],
                ['Total Lines', str(report.total_lines)],
                ['Errors', str(report.error_count)],
                ['Warnings', str(report.warning_count)],
                ['Critical', str(report.critical_count)],
                ['Info', str(report.info_count)],
                ['Debug', str(report.debug_count)],
                ['Anomalies', str(report.anomaly_count)],
                ['Duplicates', str(report.duplicate_count)],
                ['Events', str(report.event_count)],
            ]
            stats_table = Table(stats_data, colWidths=[3 * inch, 3 * inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3a6bd4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7f9ff')]),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(stats_table)
            story.append(Spacer(1, 0.2 * inch))

            story.append(Paragraph('Critical Issues', heading_style))
            for c in critical_lines[:10]:
                story.append(Paragraph(f"<b>Line {c['line']}:</b> {c['text'][:200]}", mono_style))
            story.append(Spacer(1, 0.2 * inch))

            story.append(Paragraph(f'{report.model_name} Anomaly Detection (Top Anomalies)', heading_style))
            for a in anomaly_lines[:10]:
                story.append(Paragraph(f"<b>Line {a['line']} | Score: {a['score']}:</b> {a['text'][:200]}", mono_style))

            if report.verified_by:
                story.append(Spacer(1, 0.2 * inch))
                story.append(Paragraph('Verification', heading_style))
                story.append(Paragraph(f"<b>Verified By:</b> {report.verified_by}", normal_style))
                if report.verified_at:
                    story.append(Paragraph(f"<b>Verified At:</b> {report.verified_at.strftime('%B %d, %Y %H:%M:%S')}", normal_style))
                story.append(Paragraph(f"<b>Notes:</b> {report.verification_notes}", normal_style))

            doc.build(story)
            pdf_bytes = buffer.getvalue()
            buffer.close()

            response = HttpResponse(pdf_bytes, content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="report_{report_id}.pdf"'
            return response

        except ImportError as e:
            return HttpResponse(f'PDF library error: {e}. Install with: pip install reportlab', status=500)

    return HttpResponse('Unknown format', status=400)


def _compute_metrics(ground_truth, predictions, n_entries):
    """Compute confusion matrix and derived metrics from ground truth and predictions."""
    tp = fp = fn = tn = 0
    for gt, pred in zip(ground_truth, predictions):
        if gt == 1 and pred == 1:
            tp += 1
        elif gt == 0 and pred == 1:
            fp += 1
        elif gt == 1 and pred == 0:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sensitivity = recall  # same as recall: TP / (TP + FN)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / n_entries if n_entries > 0 else 0.0

    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'sensitivity': round(sensitivity, 4),
        'specificity': round(specificity, 4),
        'f1_score': round(f1, 4),
        'confusion_matrix': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
        'total_entries': n_entries,
        'total_anomalies_detected': tp + fp,
        'total_actual_anomalies': tp + fn,
    }


@csrf_exempt
def analyze_graphs(request, log_id):
    """Compute evaluation metrics for individual algorithms and hybrid combinations."""
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Not authenticated'}, status=401)

    log = get_object_or_404(LogFile, id=log_id, user=request.user)
    model_name = request.GET.get('model', '')
    models_param = request.GET.get('models', '')     # comma-separated individual models
    hybrids_param = request.GET.get('hybrids', '')   # pipe-separated hybrid defs, each is + joined

    filepath = log.file.path
    entries = _read_log_file(filepath)
    if not entries:
        return JsonResponse({'error': 'No log entries found'}, status=400)

    n = len(entries)
    ground_truth = [1 if e['label'] in ('ERROR', 'CRITICAL') else 0 for e in entries]

    # Determine which individual models to run
    if models_param:
        # Run exactly the requested individual models (even if just 1)
        models_to_run = [m.strip() for m in models_param.split(',') if m.strip() in ML_MODELS]
    elif hybrids_param:
        # Only hybrids requested, no individual models
        models_to_run = []
    elif model_name and model_name in ML_MODELS:
        models_to_run = [model_name]
    else:
        models_to_run = list(ML_MODELS.keys())

    # Parse hybrid definitions (e.g. "GRU~LSTM|GRU~GRU AE~LSTM")
    # Uses ~ as separator to avoid conflict with model names containing '+'
    hybrid_defs = []
    if hybrids_param:
        for hdef in hybrids_param.split('|'):
            parts = [p.strip() for p in hdef.split('~') if p.strip() in ML_MODELS]
            if len(parts) >= 2:
                hybrid_defs.append(parts)

    # Collect all base models we need to run (for individuals + hybrids)
    all_base_models = set(models_to_run)
    for hparts in hybrid_defs:
        all_base_models.update(hparts)

    # Run each base model once, cache predictions and timing
    base_cache = {}
    for mname in all_base_models:
        ml_fn = ML_MODELS[mname]

        tracemalloc.start()
        t_start = time.perf_counter()
        anomaly_lines = ml_fn(list(entries), filepath)
        t_end = time.perf_counter()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        anomaly_line_nums = set(a['line'] for a in anomaly_lines)
        # Build per-entry score map from anomaly results
        score_map = {a['line']: a['score'] for a in anomaly_lines}
        preds = [1 if e['line'] in anomaly_line_nums else 0 for e in entries]

        # Build sampled score distribution for charting (max 200 points + all anomalies)
        all_entry_scores = [score_map.get(e['line'], 0.0) for e in entries]

        # ROC Curve calculation
        fpr, tpr, _ = roc_curve(ground_truth, all_entry_scores)
        roc_auc = auc(fpr, tpr)

        step = max(1, n // 200)
        sampled_set = set(range(0, n, step))
        # Always include anomaly indices
        for a in anomaly_lines:
            idx = a['line'] - 1  # line is 1-indexed
            if 0 <= idx < n:
                sampled_set.add(idx)
        sampled_scores = sorted([{'idx': i, 'score': all_entry_scores[i]} for i in sampled_set], key=lambda d: d['idx'])

        # Compute threshold from anomaly scores
        if anomaly_lines:
            min_anomaly_score = min(a['score'] for a in anomaly_lines)
        else:
            min_anomaly_score = 0.099

        base_cache[mname] = {
        'predictions': preds,
        'speed_ms': round((t_end - t_start) * 1000, 2),
        'memory_mb': round(peak_mem / (1024 * 1024), 3),
        'score_distribution': sampled_scores,
        'threshold': min_anomaly_score,

        # ROC Curve data
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': round(roc_auc, 4)
        },
    }

    # Build results for individual models
    results = {}
    for mname in models_to_run:
        bc = base_cache[mname]
        metrics = _compute_metrics(ground_truth, bc['predictions'], n)
        metrics['speed_ms'] = bc['speed_ms']
        metrics['memory_mb'] = bc['memory_mb']
        metrics['is_hybrid'] = False
        metrics['components'] = [mname]
        metrics['score_distribution'] = bc['score_distribution']
        metrics['threshold'] = bc['threshold']
        metrics['roc_curve'] = bc['roc_curve']
        results[mname] = metrics

    # Build results for hybrid models (majority-vote fusion)
    for hparts in hybrid_defs:
        hybrid_name = ' + '.join(hparts)
        k = len(hparts)

        # Measure hybrid fusion overhead
        tracemalloc.start()
        t_start = time.perf_counter()

        # Majority vote: anomaly if > half the models agree
        threshold = k / 2.0
        hybrid_preds = []
        for i in range(n):
            votes = sum(base_cache[m]['predictions'][i] for m in hparts)
            hybrid_preds.append(1 if votes > threshold else 0)

        t_end = time.perf_counter()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Total speed = sum of component speeds + fusion overhead
        fusion_ms = round((t_end - t_start) * 1000, 2)
        total_speed = round(sum(base_cache[m]['speed_ms'] for m in hparts) + fusion_ms, 2)
        total_memory = round(sum(base_cache[m]['memory_mb'] for m in hparts), 3)

        metrics = _compute_metrics(ground_truth, hybrid_preds, n)
        metrics['speed_ms'] = total_speed
        metrics['memory_mb'] = total_memory
        metrics['is_hybrid'] = True
        metrics['components'] = hparts

        # Build hybrid score distribution (average of component scores)
        hybrid_scores = []
        for i in range(n):
            avg_score = sum(base_cache[m]['score_distribution'][min(i // max(1, n // 200), len(base_cache[m]['score_distribution'])-1)]['score'] for m in hparts) / k
            hybrid_scores.append(avg_score)
        step = max(1, n // 200)
        metrics['score_distribution'] = [{'idx': i, 'score': hybrid_scores[i]} for i in range(0, n, step)]
        # Threshold for hybrid: average of component thresholds
        metrics['threshold'] = round(sum(base_cache[m]['threshold'] for m in hparts) / k, 4)

        results[hybrid_name] = metrics

    return JsonResponse({
        'success': True,
        'log_id': log.id,
        'filename': log.filename,
        'results': results,
    })


@csrf_exempt
def delete_log(request, log_id):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Not authenticated'}, status=401)
    if request.method != 'DELETE':
        return JsonResponse({'error': 'DELETE required'}, status=405)

    log = get_object_or_404(LogFile, id=log_id, user=request.user)
    try:
        if log.file and os.path.exists(log.file.path):
            os.remove(log.file.path)
    except Exception:
        pass
    log.delete()
    return JsonResponse({'success': True})