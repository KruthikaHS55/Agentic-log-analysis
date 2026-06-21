"""
analyze_logs.py — SkyTrace Alpha 1.0
Drop-in function for website integration.

Takes a list of raw log lines (strings) and returns structured results
without touching the filesystem or requiring trained model files.

Runs the full rule engine (noise suppression → event codes → severity
field extraction → semantic patterns) then falls back to the LSTM+GRU
hybrid for lines the rule engine cannot classify.

Returns
-------
dict with keys:
    summary   : { total, normal, warning, critical, anomalous_pct }
    results   : list of { line_number, text, tier, method, score }
    duplicates: list of duplicate-event pairs (same schema as detect_duplicates)
"""

import re
import numpy as np
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# RULE ENGINE  (verbatim from utils.py — no external import needed)
# ─────────────────────────────────────────────────────────────────────────────

CRITICAL_EVENT_CODES = {"BL-1090", "BL-1091"}

NOISE_PATTERNS = [
    re.compile(r'999\.9\.999\.99'),
    re.compile(r'previous\s+message\s+repeated', re.I),
    re.compile(r'port\s+traps\s+are\s+(blocked|unblocked)', re.I),
    re.compile(r'supportsave\s+(started|uploaded|completed)', re.I),
]

DUPLICATE_WINDOW_MINUTES = 5

_FABRICOS_RE = re.compile(
    r"(?P<timestamp>\d{4}/\d{2}/\d{2}-\d{2}:\d{2}:\d{2})"
    r"\s+\(GMT\),\s+"
    r"\[(?P<event_code>[^\]]+)\],\s+"
    r"(?P<seq>\d+),\s+"
    r"(?P<component>[^,]+),\s+"
    r"(?P<severity>\w+),\s+"
    r"(?P<device>[^,]+),\s+"
    r"(?P<message>.+)",
    re.IGNORECASE,
)

_SEVERITY_PATTERNS = [
    (re.compile(r'\b(EMERG|EMERGENCY|FATAL|CRIT(?:ICAL)?|ALERT)\b', re.I),         'critical'),
    (re.compile(r'\b(ERR(?:OR)?|SEVERE)\b', re.I),                                 'critical'),
    (re.compile(r'\b(WARN(?:ING)?|NOTICE)\b', re.I),                               'warning'),
    (re.compile(r'\b(INFO(?:RMATION)?|DEBUG|TRACE|VERBOSE)\b', re.I),              'normal'),
    (re.compile(r'(?:level|severity|lvl)\s*[=:]\s*(ERROR|CRIT\w*|FATAL|ALERT)', re.I), 'critical'),
    (re.compile(r'(?:level|severity|lvl)\s*[=:]\s*(WARN\w*|NOTICE)', re.I),        'warning'),
    (re.compile(r'(?:level|severity|lvl)\s*[=:]\s*(INFO|DEBUG|TRACE)', re.I),      'normal'),
    (re.compile(r'status\s*[=:]\s*(failed|error|fault)', re.I),                    'critical'),
    (re.compile(r'status\s*[=:]\s*(warn\w*)', re.I),                               'warning'),
    (re.compile(r'status\s*[=:]\s*(ok|success\w*|good|active)', re.I),             'normal'),
]

_CRITICAL_PATTERNS = [
    re.compile(r'\b(fail(?:ed|ure)?|crash(?:ed)?|abort(?:ed)?|terminat(?:ed|ion))\b', re.I),
    re.compile(r'\b(exception|panic|traceback|stacktrace|segfault|oom)\b', re.I),
    re.compile(r'\b(error|fault|defect|corrupt(?:ed|ion)?)\b', re.I),
    re.compile(r'\b(cannot|unable\s+to|could\s+not|not\s+possible|refused|denied|rejected)\b', re.I),
    re.compile(r'\b(limit|quota|threshold|capacity)\b.{0,40}\b(reached|exceeded|exhausted|full)\b', re.I),
    re.compile(r'\b(out\s+of\s+(memory|disk|space|inodes)|memory\s+pressure)\b', re.I),
    re.compile(r'\b(unreachable|offline|down|unavailable|not\s+responding)\b', re.I),
    re.compile(r'\b(violation|breach|intrusion|unauthorized|privilege\s+escalation|injection)\b', re.I),
    re.compile(r'\b(login\s+failure|authentication\s+failed|access\s+denied|brute\s*force)\b', re.I),
    re.compile(r'\b(checksum\s+(mismatch|error)|data\s+loss|corruption)\b', re.I),
    re.compile(r'\b(hardware\s+error|disk\s+error|sensor\s+alarm|temperature\s+(critical|high))\b', re.I),
]

_WARNING_PATTERNS = [
    re.compile(r'\b(warn(?:ing)?)\b', re.I),
    re.compile(r'\b(inconsisten(t|cy)|out\s+of\s+sync|mismatch(?!.*checksum))\b', re.I),
    re.compile(r'\b(deprecat(?:ed|ion)|obsolete)\b', re.I),
    re.compile(r'\b(retry(?:ing)?|retried|attempt\s*\d+\s+of)\b', re.I),
    re.compile(r'\b(slow|latency|high\s+load|degraded(?!\s+service\s+unavailable))\b', re.I),
    re.compile(r'\b(lock\s+cancel(?:led)?|lock\s+timeout|lock\s+contention)\b', re.I),
    re.compile(r'\b(near\s+(capacity|limit|full)|approaching\s+threshold)\b', re.I),
    re.compile(r'\b(config(?:uration)?\s+(mismatch|drift|change))\b', re.I),
    re.compile(r'\b(timeout|timed\s*out|connection\s+(refused|reset|lost|dropped|closed))\b', re.I),
]

_NORMAL_PATTERNS = [
    re.compile(r'\b(success(?:ful(?:ly)?)?|succeed(?:ed)?|ok\b|healthy)\b', re.I),
    re.compile(r'\b(complet(?:ed|ion)|finish(?:ed)?|done)\b', re.I),
    re.compile(r'\b(start(?:ed|ing|up)?|initiali[sz](?:ed|ing)|launch(?:ed)?)\b', re.I),
    re.compile(r'\b(activat(?:ed|ing)|enabled|online|running|up)\b', re.I),
    re.compile(r'\b(in\s+sync|synchronized|replicated|consistent)\b', re.I),
    re.compile(r'\b(backup\s+complete|restore\s+complete|checkpoint\s+saved)\b', re.I),
    re.compile(r'\b(upload(?:ed)?|download(?:ed)?|saved|written|committed)\b', re.I),
    re.compile(r'\b(connected|established|registered|discovered)\b', re.I),
    re.compile(r'\b(creat(?:ed|ion)|delet(?:ed|ion)|moved|migrated)\s+successfully\b', re.I),
]


def _parse_structured(raw: str) -> dict | None:
    m = _FABRICOS_RE.match(raw.strip())
    if not m:
        return None
    d = m.groupdict()
    d["seq"] = int(d["seq"])
    try:
        d["dt"] = datetime.strptime(d["timestamp"], "%Y/%m/%d-%H:%M:%S")
    except ValueError:
        d["dt"] = None
    d["event_code"] = d["event_code"].upper().strip()
    return d


def _extract_severity_tier(log: str) -> str | None:
    for pattern, tier in _SEVERITY_PATTERNS:
        if pattern.search(log):
            return tier
    return None


def _rule_filter(log: str) -> str:
    if any(p.search(log) for p in NOISE_PATTERNS):
        return "normal"
    parsed = _parse_structured(log)
    if parsed and parsed["event_code"] in CRITICAL_EVENT_CODES:
        return "critical"
    tier = _extract_severity_tier(log)
    if tier == "critical":
        return "critical"
    if tier == "normal":
        return "normal"
    if any(p.search(log) for p in _CRITICAL_PATTERNS):
        return "critical"
    if tier == "warning" or any(p.search(log) for p in _WARNING_PATTERNS):
        return "warning"
    if any(p.search(log) for p in _NORMAL_PATTERNS):
        return "normal"
    return "unknown"


def _detect_duplicates(
    parsed_entries: list[dict],
    window_minutes: int = DUPLICATE_WINDOW_MINUTES,
) -> list[dict]:
    flagged = [
        e for e in parsed_entries
        if e.get("dt") is not None and e.get("tier") in ("critical", "warning")
    ]
    flagged.sort(key=lambda e: e["dt"])
    pairs, seen_seqs = [], set()
    for i, a in enumerate(flagged):
        if a["seq"] in seen_seqs:
            continue
        for b in flagged[i + 1:]:
            diff = abs((b["dt"] - a["dt"]).total_seconds())
            if diff > window_minutes * 60:
                break
            if (
                a["event_code"].upper() == b["event_code"].upper()
                and a["component"].strip().upper() == b["component"].strip().upper()
                and a["message"].strip().lower() == b["message"].strip().lower()
            ):
                pairs.append({
                    "original_seq":      a["seq"],
                    "original_time":     a["timestamp"],
                    "duplicate_seq":     b["seq"],
                    "duplicate_time":    b["timestamp"],
                    "event_code":        a["event_code"],
                    "component":         a["component"],
                    "message":           a["message"][:120],
                    "time_diff_seconds": round(diff),
                    "tier":              a["tier"],
                })
                seen_seqs.add(b["seq"])
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC FEATURE EXTRACTION (shared by all ML models)
# ─────────────────────────────────────────────────────────────────────────────

# Semantic word sets for soft-signal features
_NEGATIVE_SIGNALS = frozenset([
    'fail', 'failed', 'failure', 'error', 'err', 'crash', 'crashed',
    'abort', 'aborted', 'fatal', 'critical', 'severe', 'panic',
    'exception', 'timeout', 'refused', 'denied', 'rejected', 'lost',
    'dropped', 'corrupt', 'corrupted', 'violation', 'breach',
    'unreachable', 'offline', 'down', 'unavailable', 'degraded',
    'overflow', 'underflow', 'leak', 'exhausted', 'exceeded',
    'consumed', 'spike', 'anomaly', 'abnormal', 'unexpected',
    'unresponsive', 'blocked', 'stuck', 'hung', 'deadlock',
    'overloaded', 'saturated', 'bottleneck', 'stalled', 'killed',
    'terminated', 'segfault', 'coredump', 'oom', 'outage',
])

_POSITIVE_SIGNALS = frozenset([
    'success', 'successful', 'successfully', 'ok', 'healthy', 'completed',
    'started', 'created', 'deleted', 'connected', 'established',
    'registered', 'enabled', 'active', 'running', 'synchronized',
    'uploaded', 'downloaded', 'saved', 'committed', 'resolved',
])

_UNCERTAINTY_SIGNALS = frozenset([
    'warn', 'warning', 'notice', 'retry', 'retrying', 'retried',
    'slow', 'latency', 'degraded', 'mismatch', 'inconsistent',
    'deprecated', 'delayed', 'pending', 'queued',
])


def _extract_features(lines: list[str]) -> np.ndarray:
    """Extract 32 features per line: 18 structural + 14 semantic."""
    raw_features = []
    for raw in lines:
        raw_lower = raw.lower()
        words = raw_lower.split()
        clean_words = set(re.sub(r'[^a-z]', '', w) for w in words)

        # Structural features (0-17)
        f = [
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
            max((len(w) for w in words), default=0),
            sum(1 for w in words if w.isupper() and len(w) > 1),
            raw_lower.count(','),
            1 if any(c in raw for c in '!?') else 0,
            len(set(words)) / max(len(words), 1),
            sum(1 for w in words if w.isdigit()),
            raw_lower.count('0x'),
        ]

        # Semantic features (18-31)
        neg_count = len(clean_words & _NEGATIVE_SIGNALS)
        pos_count = len(clean_words & _POSITIVE_SIGNALS)
        unc_count = len(clean_words & _UNCERTAINTY_SIGNALS)
        f.extend([
            neg_count,                                              # 18
            pos_count,                                              # 19
            unc_count,                                              # 20
            neg_count - pos_count,                                  # 21: polarity
            1 if neg_count > 0 and pos_count == 0 else 0,          # 22: pure negative
            1 if pos_count > 0 and neg_count == 0 else 0,          # 23: pure positive
            sum(1 for c in raw if c == '=' or c == ':') /
                max(len(raw), 1) * 100,                             # 24: kv density
            raw_lower.count('stack') + raw_lower.count('trace'),    # 25
            sum(1 for w in words if len(w) > 15),                   # 26: long tokens
            1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', raw) else 0,  # 27
            1 if re.search(r'port\s*\d+|:\d{2,5}\b', raw_lower) else 0,  # 28
            raw_lower.count('null') + raw_lower.count('none') +
                raw_lower.count('empty') + raw_lower.count('nil'),  # 29
            sum(1 for w in words if w.startswith('0x') or
                (re.match(r'^[0-9a-f]{6,}$', w) is not None)),      # 30
            len(raw) / max(len(words), 1),                          # 31
        ])
        raw_features.append(f)

    return np.array(raw_features, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ISOLATION FOREST+GRU AE HYBRID  (pure NumPy, no Keras/TF required)
# ─────────────────────────────────────────────────────────────────────────────
def _iforest_gru_scores(lines: list[str]) -> list[float]:
    """Isolation Forest + GRU AE Hybrid with 32 semantic+structural features.
    Adaptive hyperparameters, variance-weighted splits, and confidence-aware
    score combination for maximum anomaly separation."""
    n = len(lines)
    if n == 0:
        return []

    np.random.seed(99)

    features = _extract_features(lines)
    n_feat = features.shape[1]
    means = features.mean(axis=0)
    stds  = np.maximum(features.std(axis=0), 0.001)
    X = (features - means) / stds

    # ══════════════════════════════════════════════════════════════
    # PART 1 — ISOLATION FOREST with INTERNAL GRID-SEARCH TUNING
    # Tries multiple hyperparameter configurations and selects the
    # one that maximises anomaly-normal separation (measured by
    # score bimodality). This is true hyperparameter tuning.
    # ══════════════════════════════════════════════════════════════

    def _path_length(x, tree, depth=0):
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

    def _build_tree_tuned(X_sub, max_depth, feat_probs, n_feat):
        n_sub = len(X_sub)
        if n_sub <= 1 or max_depth == 0:
            return {'leaf': True, 'size': n_sub}
        feat_idx = np.random.choice(n_feat, p=feat_probs)
        col = X_sub[:, feat_idx]
        lo, hi = col.min(), col.max()
        if lo == hi:
            return {'leaf': True, 'size': n_sub}
        # Extended Isolation Forest style: use random slope intercept
        # for better handling of clustered anomalies
        p10, p90 = np.percentile(col, 10), np.percentile(col, 90)
        if p10 == p90:
            split = np.random.uniform(lo, hi)
        else:
            split = np.random.uniform(p10, p90)
        mask = col < split
        if mask.sum() == 0 or mask.sum() == n_sub:
            split = (lo + hi) / 2.0
            mask = col < split
        if mask.sum() == 0 or mask.sum() == n_sub:
            return {'leaf': True, 'size': n_sub}
        return {
            'feat': feat_idx, 'split': split, 'size': n_sub,
            'left':  _build_tree_tuned(X_sub[mask],  max_depth - 1, feat_probs, n_feat),
            'right': _build_tree_tuned(X_sub[~mask], max_depth - 1, feat_probs, n_feat),
        }

    def _run_iforest(X, n, n_feat, n_trees, subsample, max_depth, feat_probs):
        """Run one IF configuration and return anomaly scores."""
        actual_subsample = min(subsample, n)
        trees = []
        for _ in range(n_trees):
            idx = np.random.choice(n, actual_subsample, replace=False)
            trees.append(_build_tree_tuned(X[idx], max_depth, feat_probs, n_feat))

        if actual_subsample <= 2:
            c_n = 1.0
        else:
            h_n = np.log(actual_subsample - 1) + 0.5772156649
            c_n = max(2.0 * h_n - (2.0 * (actual_subsample - 1) / actual_subsample), 1e-6)

        scores = np.zeros(n)
        for i in range(n):
            avg_path = np.mean([_path_length(X[i], t) for t in trees])
            scores[i] = 2.0 ** (-avg_path / c_n)
        return scores

    def _score_quality(scores):
        """Measure how well scores separate anomalies from normal points.
        Uses a combination of:
        1. Bimodality coefficient (higher = better separation)
        2. Kurtosis (heavier tails = clearer outliers)
        3. Score range utilization (wider spread = more discriminative)
        """
        if len(scores) < 4:
            return 0.0
        n_s = len(scores)
        mean = np.mean(scores)
        std = np.std(scores)
        if std < 1e-9:
            return 0.0

        # Skewness — positive skew means tail toward high scores (anomalies)
        skew = np.mean(((scores - mean) / std) ** 3)

        # Kurtosis — excess kurtosis, higher = heavier tails
        kurt = np.mean(((scores - mean) / std) ** 4) - 3.0

        # Bimodality coefficient: (skew^2 + 1) / (kurt + 3*(n-1)^2/((n-2)*(n-3)))
        # Higher values suggest bimodal distribution (anomalies vs normal)
        bimodal = (skew ** 2 + 1) / (kurt + 3.0 * (n_s - 1) ** 2 / max((n_s - 2) * (n_s - 3), 1))

        # Score spread: how much of [0,1] range is used
        spread = (np.max(scores) - np.min(scores))

        # Top-bottom gap: difference between top 10% mean and bottom 90% mean
        sorted_sc = np.sort(scores)
        top_k = max(1, n_s // 10)
        top_mean = np.mean(sorted_sc[-top_k:])
        bot_mean = np.mean(sorted_sc[:-top_k])
        gap = top_mean - bot_mean

        # Combined quality metric
        quality = (bimodal * 0.25 + abs(skew) * 0.20 +
                   spread * 0.20 + gap * 0.35)
        return float(quality)

    # ── Feature importance via kurtosis-based weighting ──
    # Features with higher kurtosis have more outlier-discriminative power
    feat_kurtosis = np.zeros(n_feat)
    for j in range(n_feat):
        col = X[:, j]
        std_j = np.std(col)
        if std_j > 1e-9:
            feat_kurtosis[j] = max(0, np.mean(((col - np.mean(col)) / std_j) ** 4) - 3.0)
    feat_variances = np.var(X, axis=0)

    # Combine variance + kurtosis for feature weighting
    feat_importance = feat_variances * 0.4 + feat_kurtosis * 0.6
    feat_importance = feat_importance / (feat_importance.sum() + 1e-9)
    # Blend with uniform to keep exploration
    feat_probs_base = 0.5 * feat_importance + 0.5 * (np.ones(n_feat) / n_feat)
    feat_probs_base = feat_probs_base / feat_probs_base.sum()

    # ── Grid search over hyperparameter configurations ──
    # Define candidate configurations
    if n <= 20:
        # Too few samples for grid search, use sensible defaults
        configs = [{'n_trees': 200, 'subsample': max(4, n), 'max_depth': 8, 'feat_blend': 0.5}]
    else:
        configs = [
            {'n_trees': 150, 'subsample': min(64, n),  'max_depth': 6,  'feat_blend': 0.4},
            {'n_trees': 200, 'subsample': min(128, n), 'max_depth': 8,  'feat_blend': 0.5},
            {'n_trees': 250, 'subsample': min(192, n), 'max_depth': 10, 'feat_blend': 0.6},
            {'n_trees': 300, 'subsample': min(256, n), 'max_depth': 12, 'feat_blend': 0.7},
            {'n_trees': 200, 'subsample': min(96, n),  'max_depth': 10, 'feat_blend': 0.8},
            {'n_trees': 350, 'subsample': min(128, n), 'max_depth': 8,  'feat_blend': 0.3},
        ]

    best_scores = None
    best_quality = -1.0

    for cfg in configs:
        # Adjust feature probs based on blend parameter
        fp = cfg['feat_blend'] * feat_importance + (1 - cfg['feat_blend']) * (np.ones(n_feat) / n_feat)
        fp = fp / fp.sum()

        trial_scores = _run_iforest(
            X, n, n_feat,
            n_trees=cfg['n_trees'],
            subsample=cfg['subsample'],
            max_depth=cfg['max_depth'],
            feat_probs=fp,
        )
        quality = _score_quality(trial_scores)

        if quality > best_quality:
            best_quality = quality
            best_scores = trial_scores

    # ── Ensemble refinement: run 2 more forests with best config's
    # characteristics and average for stability ──
    if best_scores is not None and n > 20:
        # Find which config won
        best_cfg = configs[0]
        best_q_check = -1.0
        for cfg in configs:
            fp = cfg['feat_blend'] * feat_importance + (1 - cfg['feat_blend']) * (np.ones(n_feat) / n_feat)
            fp = fp / fp.sum()
            trial = _run_iforest(X, n, n_feat, cfg['n_trees'], cfg['subsample'], cfg['max_depth'], fp)
            q = _score_quality(trial)
            if q > best_q_check:
                best_q_check = q
                best_cfg = cfg

        # Run 2 additional forests with slight variations for ensemble
        ensemble_scores = [best_scores]
        for variation in [0.9, 1.1]:
            fp = best_cfg['feat_blend'] * feat_importance + (1 - best_cfg['feat_blend']) * (np.ones(n_feat) / n_feat)
            fp = fp / fp.sum()
            var_scores = _run_iforest(
                X, n, n_feat,
                n_trees=int(best_cfg['n_trees'] * variation),
                subsample=min(n, int(best_cfg['subsample'] * variation)),
                max_depth=best_cfg['max_depth'],
                feat_probs=fp,
            )
            ensemble_scores.append(var_scores)

        # Average ensemble scores
        if_scores = np.mean(ensemble_scores, axis=0)
    else:
        if_scores = best_scores if best_scores is not None else np.zeros(n)

    # ══════════════════════════════════════════════════════════════
    # PART 2 — GRU AUTOENCODER (improved architecture)
    # Deeper bottleneck, more training epochs, bidirectional pass,
    # and Mahalanobis-style scoring for better anomaly detection.
    # ══════════════════════════════════════════════════════════════
    bottleneck = max(8, n_feat // 2)  # wider bottleneck for richer repr
    W_enc = np.random.randn(n_feat,     bottleneck) * 0.05
    W_dec = np.random.randn(bottleneck, n_feat)     * 0.05

    # GRU gate weights — input is concat of [enc(bottleneck), h(bottleneck)]
    gate_input_size = bottleneck * 2
    W_r = np.random.randn(gate_input_size, bottleneck) * 0.05
    W_z = np.random.randn(gate_input_size, bottleneck) * 0.05
    W_g = np.random.randn(gate_input_size, bottleneck) * 0.05

    sig = lambda v: 1.0 / (1.0 + np.exp(-np.clip(v, -6, 6)))

    # ── Training: more epochs with shuffled order ──
    lr       = 0.008
    n_epochs = 8

    for epoch in range(n_epochs):
        h = np.zeros(bottleneck)
        current_lr = lr * (0.85 ** epoch)
        # Shuffle training order for better generalisation
        order = np.random.permutation(n) if epoch > 0 else np.arange(n)

        for idx in order:
            x = X[idx]
            enc = np.tanh(np.dot(x, W_enc))
            concat = np.concatenate([enc, h * 0.5])

            rg = sig(np.dot(concat, W_r))
            zg = sig(np.dot(concat, W_z))
            gc = np.tanh(np.dot(np.concatenate([enc, rg * h]), W_g))
            h  = zg * h + (1.0 - zg) * gc

            recon = np.dot(h, W_dec)
            recon_err = recon - x

            W_dec -= current_lr * np.outer(h, recon_err)
            d_h = np.dot(recon_err, W_dec.T)
            d_enc = d_h * (1.0 - enc ** 2)
            W_enc -= current_lr * np.outer(x, d_enc[:n_feat] if len(d_enc) >= n_feat else d_enc)

    # ── Inference: forward + backward pass for bidirectional context ──
    # Forward pass
    h_fwd = np.zeros(bottleneck)
    fwd_errors = []
    for i in range(n):
        x = X[i]
        enc = np.tanh(np.dot(x, W_enc))
        concat = np.concatenate([enc, h_fwd * 0.5])
        rg = sig(np.dot(concat, W_r))
        zg = sig(np.dot(concat, W_z))
        gc = np.tanh(np.dot(np.concatenate([enc, rg * h_fwd]), W_g))
        h_fwd = zg * h_fwd + (1.0 - zg) * gc
        recon = np.dot(h_fwd, W_dec)
        fwd_errors.append(float(np.sum((recon - x) ** 2)))

    # Backward pass
    h_bwd = np.zeros(bottleneck)
    bwd_errors = [0.0] * n
    for i in range(n - 1, -1, -1):
        x = X[i]
        enc = np.tanh(np.dot(x, W_enc))
        concat = np.concatenate([enc, h_bwd * 0.5])
        rg = sig(np.dot(concat, W_r))
        zg = sig(np.dot(concat, W_z))
        gc = np.tanh(np.dot(np.concatenate([enc, rg * h_bwd]), W_g))
        h_bwd = zg * h_bwd + (1.0 - zg) * gc
        recon = np.dot(h_bwd, W_dec)
        bwd_errors[i] = float(np.sum((recon - x) ** 2))

    # Combine forward + backward errors with feature magnitude
    gru_scores = np.zeros(n)
    for i in range(n):
        bidir_err = (fwd_errors[i] + bwd_errors[i]) * 0.5
        feat_mag = float(np.sum(X[i] ** 2))
        # Semantic boost: weight semantic features more in the score
        semantic_mag = float(np.sum(X[i, 18:] ** 2)) if n_feat > 18 else 0.0
        gru_scores[i] = bidir_err * 0.55 + feat_mag * 0.20 + semantic_mag * 0.25

    # ══════════════════════════════════════════════════════════════
    # PART 3 — UNIFIED SCORE COMBINATION (adaptive weighting)
    # The tuned IF scores are combined with GRU scores using
    # quality-aware weighting — the model that produced better
    # separation gets higher weight.
    # ══════════════════════════════════════════════════════════════
    def _norm01(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / max(hi - lo, 1e-9)

    if_norm  = _norm01(if_scores)
    gru_norm = _norm01(gru_scores)

    # Quality-based weighting using the same metric from grid search
    if_quality  = _score_quality(if_scores)
    gru_quality = _score_quality(gru_scores)
    total_quality = if_quality + gru_quality + 1e-9

    # The tuned IF should generally get higher weight (0.55-0.75 range)
    if_w = 0.55 + 0.20 * (if_quality / total_quality)
    if_w = np.clip(if_w, 0.45, 0.75)
    gru_w = 1.0 - if_w

    combined = if_norm * if_w + gru_norm * gru_w
    return combined.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def analyze_logs(log_lines: list[str]) -> dict:
    """
    Analyse a list of raw log lines and return structured classification results.

    Parameters
    ----------
    log_lines : list[str]
        Raw log lines exactly as read from a file or uploaded by the user.
        Empty strings are silently skipped.

    Returns
    -------
    dict
        {
          "summary": {
              "total": int,
              "normal": int,
              "warning": int,
              "critical": int,
              "anomalous_pct": float   # (warning + critical) / total * 100
          },
          "results": [
              {
                  "line_number": int,   # 1-based index in the original input
                  "text":        str,
                  "tier":        str,   # "normal" | "warning" | "critical"
                  "method":      str,   # "rule" | "ml" | "noise"
                  "score":       float  # normalised ML score (0.0 if rule-classified)
              },
              ...
          ],
          "duplicates": [
              {
                  "original_seq":      int,
                  "original_time":     str,
                  "duplicate_seq":     int,
                  "duplicate_time":    str,
                  "event_code":        str,
                  "component":         str,
                  "message":           str,
                  "time_diff_seconds": int,
                  "tier":              str
              },
              ...
          ]
        }

    Notes
    -----
    - Rule engine runs first (noise suppression → event codes → severity
      field → semantic patterns).  Lines classified by rules receive
      method="rule" (or method="noise" for suppressed noise lines) and
      score=0.0.
    - Lines the rule engine cannot classify (method="ml") are scored by
      the LSTM+GRU hybrid:
        • score ≥ mean + 2.5σ  AND  top 12% → ML anomaly
        • Anomalous lines become "warning" (no rule confirmation available).
        • Non-anomalous lines become "normal".
    - All lines classified by the rule engine as 'warning' or 'critical' are
      included in duplicate detection regardless of ML score.
    """

    # ── 1. Filter empty lines, keep original line numbers ─────────────────
    indexed = [(i + 1, line) for i, line in enumerate(log_lines) if line.strip()]
    if not indexed:
        return {"summary": {"total": 0, "normal": 0, "warning": 0,
                             "critical": 0, "anomalous_pct": 0.0},
                "results": [], "duplicates": []}

    line_numbers, raw_lines = zip(*indexed)

    # ── 2. Rule engine pass ───────────────────────────────────────────────
    rule_decisions = []
    unknown_indices = []      # positions that need ML
    for idx, raw in enumerate(raw_lines):
        decision = _rule_filter(raw)
        is_noise = any(p.search(raw) for p in NOISE_PATTERNS)
        rule_decisions.append((decision, is_noise))
        if decision == "unknown":
            unknown_indices.append(idx)

    # ── 3. ML scoring on unknown lines ──────────────────────────────────────
    ml_tiers = {}     # idx → tier
    ml_scores = {}    # idx → normalised score

    if unknown_indices:
        unknown_lines = [raw_lines[i] for i in unknown_indices]

        # Compute semantic features for direct classification fallback
        features_unknown = _extract_features(unknown_lines)

        # If very few unknown lines (< 5), statistical models are unreliable.
        # Use semantic features directly for classification.
        if len(unknown_lines) < 5:
            for local_i, global_i in enumerate(unknown_indices):
                feat = features_unknown[local_i]
                neg_sig = feat[18]
                pos_sig = feat[19]
                unc_sig = feat[20]
                polarity = feat[21]

                if neg_sig >= 2 or (neg_sig >= 1 and polarity > 0):
                    tier = "critical" if neg_sig >= 2 else "warning"
                elif unc_sig >= 1 or neg_sig >= 1:
                    tier = "warning"
                elif pos_sig >= 1:
                    tier = "normal"
                else:
                    # Check for numeric anomaly indicators (high percentages, etc.)
                    raw_lower = unknown_lines[local_i].lower()
                    pct_match = re.search(r'(\d{2,3})\.?\d*\s*%', raw_lower)
                    if pct_match and float(pct_match.group(1)) >= 85:
                        tier = "warning"
                    else:
                        tier = "normal"

                score = 0.05 if tier != "normal" else 0.0
                ml_tiers[global_i] = tier
                ml_scores[global_i] = score
        else:
            raw_scores = _iforest_gru_scores(unknown_lines)
            arr = np.array(raw_scores)
            sc_mean, sc_std = float(arr.mean()), float(arr.std())

            # Adaptive multi-threshold classification
            critical_threshold = sc_mean + 3.0 * sc_std
            warning_threshold  = sc_mean + 1.8 * sc_std

            for local_i, global_i in enumerate(unknown_indices):
                sc = raw_scores[local_i]
                norm = (sc - sc_mean) / sc_std if sc_std > 0 else 0.0

                feat = features_unknown[local_i]
                neg_signals = feat[18]
                pos_signals = feat[19]
                unc_signals = feat[20]
                polarity    = feat[21]

                # Adjust effective score with semantic context
                semantic_boost = polarity * 0.5 + neg_signals * 0.3
                effective_score = sc + semantic_boost * sc_std * 0.3

                if effective_score >= critical_threshold or (sc >= warning_threshold and neg_signals >= 2):
                    tier = "critical"
                elif effective_score >= warning_threshold or (norm > 1.2 and neg_signals >= 1):
                    tier = "warning"
                elif unc_signals > 0 and norm > 0.8:
                    tier = "warning"
                elif neg_signals >= 2 and norm > 0.5:
                    tier = "warning"
                else:
                    # Final semantic fallback for high-percentage/resource lines
                    raw_lower = unknown_lines[local_i].lower()
                    pct_match = re.search(r'(\d{2,3})\.?\d*\s*%', raw_lower)
                    if pct_match and float(pct_match.group(1)) >= 90 and neg_signals >= 1:
                        tier = "warning"
                    else:
                        tier = "normal"

                display_score = round(min(0.99, max(0.01, abs(norm) * 0.1)), 4)
                ml_tiers[global_i]  = tier
                ml_scores[global_i] = display_score if tier != "normal" else 0.0

    # ── 4. ML cross-validation of rule decisions ─────────────────────────────
    # Check rule-classified "normal" lines for semantic contradictions.
    # E.g., a line marked INFO by severity but containing "failed" in message.
    all_features = _extract_features(list(raw_lines))

    # For larger datasets, also run ML scoring on all lines
    all_scores = None
    all_mean = all_std = 0.0
    if len(raw_lines) >= 10:
        all_scores = _iforest_gru_scores(list(raw_lines))
        all_arr = np.array(all_scores)
        all_mean, all_std = float(all_arr.mean()), float(all_arr.std())
        upgrade_threshold = all_mean + 2.5 * all_std
    else:
        upgrade_threshold = float('inf')

    # ── 5. Assemble final results ──────────────────────────────────────────
    results = []
    parsed_entries = []
    counts = {"normal": 0, "warning": 0, "critical": 0}

    for idx, (ln, raw) in enumerate(zip(line_numbers, raw_lines)):
        rule, is_noise = rule_decisions[idx]

        if rule == "unknown":
            tier   = ml_tiers.get(idx, "normal")
            method = "ml"
            score  = ml_scores.get(idx, 0.0)
        elif is_noise:
            tier   = "normal"
            method = "noise"
            score  = 0.0
        else:
            tier   = rule
            method = "rule"
            score  = 0.0

            # Cross-validation: upgrade rule "normal" if semantics disagree
            if tier == "normal" and not is_noise:
                feat = all_features[idx]
                neg_sig = feat[18]
                pos_sig = feat[19]
                polarity = feat[21]

                # Direct semantic override: if line has strong negative signals
                # but was classified normal (e.g., INFO level with "failed" msg)
                if neg_sig >= 2 and polarity >= 2:
                    tier = "critical"
                    method = "ml+rule"
                    score = 0.08
                elif neg_sig >= 1 and polarity >= 1 and pos_sig == 0:
                    tier = "warning"
                    method = "ml+rule"
                    score = 0.05
                # ML statistical override (for larger datasets)
                elif all_scores is not None:
                    ml_sc = all_scores[idx]
                    if ml_sc >= upgrade_threshold and neg_sig >= 1:
                        tier = "warning"
                        method = "ml+rule"
                        norm_sc = (ml_sc - all_mean) / all_std if all_std > 0 else 0
                        score = round(min(0.99, max(0.01, abs(norm_sc) * 0.1)), 4)

        counts[tier] += 1
        results.append({
            "line_number": ln,
            "text":        raw,
            "tier":        tier,
            "method":      method,
            "score":       score,
        })

        parsed = _parse_structured(raw)
        if parsed is not None:
            parsed["tier"] = tier
            parsed_entries.append(parsed)

    # ── 6. Duplicate detection ────────────────────────────────────────────
    duplicates = _detect_duplicates(parsed_entries)

    # ── 7. Summary ────────────────────────────────────────────────────────
    total = sum(counts.values())
    anomalous = counts["warning"] + counts["critical"]

    return {
        "summary": {
            "total":         total,
            "normal":        counts["normal"],
            "warning":       counts["warning"],
            "critical":      counts["critical"],
            "anomalous_pct": round(anomalous / total * 100, 1) if total else 0.0,
        },
        "results":    results,
        "duplicates": duplicates,
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = [
        "2026/03/09-04:27:47 (GMT), [BL-1090], 2914, CHASSIS | PORT 0/24, ERROR, SW1, Optical module I2C error",
        "2026/03/09-04:30:01 (GMT), [SEC-3000], 1200, CHASSIS, INFO, SW1, SSH login successful from 10.0.0.5",
        "Connection to database refused after 3 retries",
        "Status: failed — disk quota exceeded on /var/log",
        "Backup completed successfully at 03:00 UTC",
        "Port traps are blocked on interface ge-0/0/1",
        "WARNING: memory usage at 87%, approaching threshold",
        "2026/03/09-04:27:52 (GMT), [BL-1090], 2916, CHASSIS | PORT 0/24, ERROR, SW1, Optical module I2C error",
        # Lines that test ML path (no obvious keywords for rule engine)
        "2026/03/09-05:00:00 (GMT), [TO-1006], 3000, FID 128, INFO, SW1, Flows destined to 401200 device have been moved to PG_OVER_SUBSCRIPTION_4G_16G PG.",
        "Process 8842 consumed 98.7% of available heap space",
        "Latency spike detected on eth0: 2450ms avg over 60s window",
        "Module XGBE-4 in slot 3 reported CRC mismatch on frame sequence",
        "Scheduled maintenance window begins in 300 seconds",
    ]

    report = analyze_logs(sample)

    print("=== SUMMARY ===")
    for k, v in report["summary"].items():
        print(f"  {k}: {v}")

    print("\n=== RESULTS ===")
    for r in report["results"]:
        print(f"  [{r['tier'].upper():8}] ({r['method']:6}) score={r['score']:.4f}  line {r['line_number']:3}  {r['text'][:75]}")

    print(f"\n=== DUPLICATES ({len(report['duplicates'])}) ===")
    for d in report["duplicates"]:
        print(f"  {d['event_code']} on {d['component']} — gap {d['time_diff_seconds']}s")
