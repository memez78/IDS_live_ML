#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live IDS Inference – FINAL PRODUCTION (v3 patches + AGGRESSIVE OVERRIDE)
- Exact 59‑feature parity with notebook (order from feature_list.txt)
- CV features (fwd_iat_cv, bwd_iat_cv, fwd_len_cv, bwd_len_cv, iat_cv)
- Protocol flags: proto_1, proto_2, proto_6 (binary, no .0 suffix)
- HTTP interaction: http_post_ratio, http_get_rate
- Volumetric guard: port 80, pkt_rate > 100, pred==5 and no hard SQL evidence -> HTTP_Flood
- Brute Force ports override (21,22,23,25,110,139,143,445,3306,3389)
- SQLi only if sqli_hard_flag == 1 or confidence >= 0.95
- C2 low‑rate override (pkt_rate < 10, >=8 packets, no SQL keywords)
- Temporal Correlation Engine + reset_state()

FIXES applied (v2):
  FIX-1  SYN early-exit now forces pred=SYN_Flood (was whatever model said)
  FIX-2  Brute-Force override covers ALL predictions on BRUTE_FORCE_PORTS,
         not just pred∈{3,5}; port list unified with BRUTE_FORCE_PORTS constant
  FIX-3  TemporalCorrelationEngine gains reset_state() alias matching LiveIDS API;
         LiveIDS.reset_state() now also resets flow_count / alert_count correctly
  FIX-4  SQLi hard-flag sets conf_pass=True *before* other overrides can demote it,
         and is re-checked on the *final* pred rather than the original pred
  FIX-5  HTTP_Flood volumetric guard also fires when pred==4 (Brute_Force) on port 80
         with no SQL evidence, preventing false Brute_Force on web floods

FIXES applied (v3):
  FIX-6  Mirror Port Bug: Override 1 (Brute_Force) now checks BOTH src_port and
         dst_port against BRUTE_FORCE_PORTS. flow_key() uses an IP+port
         lexicographic sort, so which port ends up as dst_port is unpredictable
         when IPs differ in ordering. Without this, attacks from a high ephemeral
         port (e.g. 55482) to a management port (e.g. 23) were silently missed.
  FIX-7  Hot-path timeout reaping: _flush_timed_out() is now called at the top of
         _packet_callback (immediately after the IP check) so zombie flows are
         cleaned up before any new packet is processed. Previously it was only
         called at the end, allowing flow table bloat during sustained attacks.
  FIX-8  http_get_rate wired into Override 2: high GET rate on port 80 with no
         SQL evidence is now an additional HTTP Flood signal. Previously the value
         was computed in finalize_flow but never read by any override logic.

AGGRESSIVE VOLUMETRIC GUARD (honest fix):
  If port 80 is involved, packet rate > 100 pkt/s, the model predicts SQLi (5),
  but there is no hard forensic SQL evidence (sqli_hard == 0), then force pred=2
  (HTTP_Flood). This overrides the ML hallucination that benchmark tools like
  ApacheBench are SQL injection attacks.
"""

import os
import re
import time
import argparse
import sys
import warnings
from urllib.parse import unquote
from collections import defaultdict

import numpy as np
import joblib
from scipy import signal

from scapy.all import sniff, get_if_list, conf as scapy_conf, IP, TCP, UDP, Raw

# ------------------------------------------------------------
# TUNABLE KNOBS (do not change unless you know what you are doing)
# ------------------------------------------------------------
FLOW_TIMEOUT = 120
TIMEOUT_CHECK_INTERVAL = 5
MIN_PKTS_FOR_CLASSIFICATION = 6
SYN_EARLY_EXIT_PKTS = 3
SYN_ONLY_RATIO_THRESHOLD = 0.80
ALERT_COOLDOWN_SECS = 10
AGG_FLUSH_SECS = 3
C2_THRESHOLD = 0.55
SQLI_KEYWORD_HARD_THRESHOLD = 5
MAX_TLS_PKTS = 10
MAX_PAYLOADS = 20
HTTP_PAYLOAD_PORTS = {80}
# FIX-6: single source-of-truth; checked against BOTH ports in every flow
BRUTE_FORCE_PORTS = {21, 22, 23, 25, 110, 139, 143, 445, 3306, 3389}

# ------------------------------------------------------------
# Label & Confidence Maps
# ------------------------------------------------------------
LABEL_MAP = {
    0: "Normal",
    1: "SYN_Flood",
    2: "HTTP_Flood",
    3: "Command_Control",
    4: "Brute_Force",
    5: "SQL_Injection",
}
CONFIDENCE_THRESHOLDS = {
    "Normal": None,
    "SYN_Flood": 0.70,
    "HTTP_Flood": 0.50,
    "Command_Control": C2_THRESHOLD,
    "Brute_Force": 0.70,
    "SQL_Injection": 0.95,
}

# SQL keyword lists
SQL_KEYWORDS = [
    "select", "union", "insert", "update", "delete", "drop",
    "exec", "execute", "cast", "convert", "declare", "xp_",
    "from", "where", "having", "order", "group",
]
CLASSIC_PATTERNS = ["or 1=1", "or 1=1--", "or '1'='1", "1=1", "or 'x'='x"]
TIME_PATTERNS = ["waitfor", "sleep", "benchmark", "pg_sleep"]

# ------------------------------------------------------------
# Terminal colours (optional)
# ------------------------------------------------------------
try:
    import colorama
    colorama.init(autoreset=False)
    _COLOUR_OK = True
except ImportError:
    _COLOUR_OK = sys.stdout.isatty()

if _COLOUR_OK:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    PURPLE = "\033[95m"
    DIM = "\033[2m"
    RESET = "\033[0m"
else:
    RED = YELLOW = GREEN = CYAN = PURPLE = DIM = RESET = ""

LABEL_COLORS = {
    "Normal": GREEN,
    "SYN_Flood": RED,
    "HTTP_Flood": YELLOW,
    "Command_Control": PURPLE,
    "Brute_Force": RED,
    "SQL_Injection": CYAN,
}

# ------------------------------------------------------------
# Feature extraction helpers
# ------------------------------------------------------------
def compute_periodicity_features(iat_array):
    MIN_IATS = 8
    if len(iat_array) < MIN_IATS:
        return 0.0, 0
    iats = np.array(iat_array, dtype=float)
    iats = (iats - np.mean(iats)) / (np.std(iats) + 1e-9)
    freqs, psd = signal.periodogram(iats)
    psd_ac = psd[1:]
    if len(psd_ac) > 0:
        total_power = np.sum(psd_ac) + 1e-9
        dominant_pwr = np.max(psd_ac)
        num_peaks = int(np.sum(psd_ac > 0.1 * dominant_pwr))
    else:
        total_power, dominant_pwr, num_peaks = 1e-9, 0.0, 0
    return float(dominant_pwr / total_power), num_peaks

def extract_tls_features(flow_packets):
    feats = {"is_tls": 0}
    for pkt in flow_packets:
        if pkt.haslayer(TCP) and pkt.haslayer(Raw):
            raw = bytes(pkt[Raw].load)
            if len(raw) > 5 and raw[0] == 0x16:
                feats["is_tls"] = 1
                break
    return feats

def extract_sqli_features(payloads, dst_port=0):
    feats = {
        "sql_keyword_count": 0,
        "quote_count": 0,
        "comment_count": 0,
        "classic_injection": 0,
        "hex_encoding_ratio": 0.0,
        "time_based_indicators": 0,
        "sqli_hard_flag": 0,
    }
    if not payloads:
        return feats
    for raw in payloads:
        chunk = raw[:512]
        raw_text = chunk.decode("utf-8", errors="ignore").lower()
        hex_hits = len(re.findall(r"%[0-9a-f]{2}", raw_text))
        feats["hex_encoding_ratio"] += hex_hits / max(len(raw_text), 1)
        text = unquote(raw_text)
        feats["sql_keyword_count"] += sum(1 for kw in SQL_KEYWORDS if kw in text)
        feats["quote_count"] += text.count("'") + text.count('"')
        feats["comment_count"] += text.count("--") + text.count("/*")
        feats["classic_injection"] += int(any(p in text for p in CLASSIC_PATTERNS))
        feats["time_based_indicators"] += sum(1 for p in TIME_PATTERNS if p in text)
    if dst_port in (80, 443, 3306) and feats["sql_keyword_count"] > SQLI_KEYWORD_HARD_THRESHOLD:
        feats["sqli_hard_flag"] = 1
    return feats

def safe_stats(arr):
    if not arr:
        return 0.0, 0.0, 0.0, 0.0
    a = np.array(arr, dtype=float)
    return a.min(), a.max(), a.mean(), a.std()

def compute_payload_entropy(payloads):
    if not payloads:
        return 0.0
    all_bytes = b"".join(payloads)
    if not all_bytes:
        return 0.0
    counts = np.bincount(np.frombuffer(all_bytes, dtype=np.uint8), minlength=256)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

# ------------------------------------------------------------
# Flow handling
# ------------------------------------------------------------
def flow_key(pkt):
    if not pkt.haslayer(IP):
        return None, None
    src, dst = pkt[IP].src, pkt[IP].dst
    proto = pkt[IP].proto
    sport = dport = 0
    if TCP in pkt:
        sport, dport = pkt[TCP].sport, pkt[TCP].dport
    elif UDP in pkt:
        sport, dport = pkt[UDP].sport, pkt[UDP].dport
    if (src, sport) <= (dst, dport):
        return (src, dst, sport, dport, proto), True
    return (dst, src, dport, sport, proto), False

def new_flow():
    return {
        "start_ts": None, "last_ts": None, "prev_ts": None,
        "fwd_pkts": 0, "bwd_pkts": 0, "fwd_bytes": 0, "bwd_bytes": 0,
        "fwd_lens": [], "bwd_lens": [], "all_iats": [], "fwd_iats": [], "bwd_iats": [],
        "fwd_last_ts": None, "bwd_last_ts": None,
        "fwd_syn": 0, "bwd_syn": 0, "fwd_ack": 0, "bwd_ack": 0,
        "fwd_fin": 0, "bwd_fin": 0, "fwd_rst": 0, "bwd_rst": 0,
        "fwd_psh": 0, "bwd_psh": 0, "fwd_urg": 0, "bwd_urg": 0,
        "http_get": 0, "http_post": 0,
        "payloads": [], "tls_packets": [], "dst_ips": [],
    }

def update_flow(flow, pkt, is_fwd):
    ts = float(pkt.time)
    plen = len(pkt)
    if flow["start_ts"] is None:
        flow["start_ts"] = ts
    if flow["prev_ts"] is not None:
        flow["all_iats"].append(ts - flow["prev_ts"])
    flow["prev_ts"] = ts
    flow["last_ts"] = ts
    d = "fwd" if is_fwd else "bwd"
    flow[f"{d}_pkts"] += 1
    flow[f"{d}_bytes"] += plen
    flow[f"{d}_lens"].append(plen)
    lk = f"{d}_last_ts"
    if flow[lk] is not None:
        flow[f"{d}_iats"].append(ts - flow[lk])
    flow[lk] = ts

    if TCP in pkt:
        f = int(pkt[TCP].flags)
        for name, mask in [("syn",0x02),("ack",0x10),("fin",0x01),
                           ("rst",0x04),("psh",0x08),("urg",0x20)]:
            if f & mask:
                flow[f"{d}_{name}"] += 1

    if Raw in pkt and TCP in pkt and pkt[TCP].dport in HTTP_PAYLOAD_PORTS:
        try:
            raw = bytes(pkt[Raw].load)
            if raw[:3] == b"GET":
                flow["http_get"] += 1
            elif raw[:4] == b"POST":
                flow["http_post"] += 1
        except Exception:
            pass

    flow["dst_ips"].append(pkt[IP].dst)

    if TCP in pkt:
        dport_p, sport_p = pkt[TCP].dport, pkt[TCP].sport
        if (dport_p in HTTP_PAYLOAD_PORTS or sport_p in HTTP_PAYLOAD_PORTS) and len(flow["payloads"]) < MAX_PAYLOADS:
            if Raw in pkt:
                try:
                    flow["payloads"].append(bytes(pkt[Raw].load)[:512])
                except Exception:
                    pass

        if (dport_p == 443 or sport_p == 443) and len(flow["tls_packets"]) < MAX_TLS_PKTS:
            try:
                payload_bytes = bytes(pkt[TCP].payload)
                if len(payload_bytes) > 5 and payload_bytes[0] in (0x16, 0x17):
                    flow["tls_packets"].append(pkt)
            except Exception:
                pass

def finalize_flow(fid, flow):
    """Return a dict with ALL 59 features in the exact order expected by the model."""
    eps = 1e-6
    dur = (flow["last_ts"] - flow["start_ts"]) if flow["start_ts"] else 0.0
    tp = flow["fwd_pkts"] + flow["bwd_pkts"]
    tb = flow["fwd_bytes"] + flow["bwd_bytes"]

    fl = safe_stats(flow["fwd_lens"])
    bl = safe_stats(flow["bwd_lens"])
    ia = safe_stats(flow["all_iats"])
    fi = safe_stats(flow["fwd_iats"])
    bi = safe_stats(flow["bwd_iats"])

    total_syn = flow["fwd_syn"] + flow["bwd_syn"]
    total_ack = flow["fwd_ack"] + flow["bwd_ack"]
    total_fin = flow["fwd_fin"] + flow["bwd_fin"]
    total_rst = flow["fwd_rst"] + flow["bwd_rst"]

    src, dst, sport, dport, proto = fid

    sqli_feats = extract_sqli_features(flow["payloads"], dport)
    tls_feats = extract_tls_features(flow["tls_packets"])
    entropy = compute_payload_entropy(flow["payloads"])
    p_ratio, p_peaks = compute_periodicity_features(flow["all_iats"])

    # CV features
    fwd_iat_cv = fi[3] / (fi[2] + eps) if fi[2] > eps else 0.0
    bwd_iat_cv = bi[3] / (bi[2] + eps) if bi[2] > eps else 0.0
    fwd_len_cv = fl[3] / (fl[2] + eps) if fl[2] > eps else 0.0
    bwd_len_cv = bl[3] / (bl[2] + eps) if bl[2] > eps else 0.0
    iat_cv = ia[3] / (ia[2] + eps) if ia[2] > eps else 0.0

    # Protocol one-hot (binary)
    proto_1 = 1.0 if proto == 1 else 0.0
    proto_2 = 1.0 if proto == 2 else 0.0
    proto_6 = 1.0 if proto == 6 else 0.0

    http_post_ratio = round(flow["http_post"] / (flow["http_get"] + flow["http_post"] + eps), 4)
    http_get_rate = round((tp / (dur + eps)) * (1.0 - http_post_ratio), 4)

    return {
        "dst_port": dport,
        "duration": round(dur, 6),
        "fwd_pkts": flow["fwd_pkts"],
        "fwd_bytes": flow["fwd_bytes"],
        "bwd_bytes": flow["bwd_bytes"],
        "pkt_rate": round(tp/(dur+eps), 4),
        "byte_rate": round(tb/(dur+eps), 4),
        "fwd_pkt_len_min": fl[0],
        "fwd_pkt_len_max": fl[1],
        "fwd_pkt_len_mean": fl[2],
        "bwd_pkt_len_min": bl[0],
        "bwd_pkt_len_max": bl[1],
        "bwd_pkt_len_mean": bl[2],
        "bwd_pkt_len_std": bl[3],
        "iat_mean": ia[2],
        "iat_std": ia[3],
        "iat_min": ia[0],
        "iat_max": ia[1],
        "fwd_iat_mean": fi[2],
        "fwd_iat_std": fi[3],
        "fwd_iat_min": fi[0],
        "fwd_iat_max": fi[1],
        "bwd_iat_mean": bi[2],
        "bwd_iat_std": bi[3],
        "bwd_iat_min": bi[0],
        "bwd_iat_max": bi[1],
        "fwd_syn": flow["fwd_syn"],
        "bwd_syn": flow["bwd_syn"],
        "fwd_fin": flow["fwd_fin"],
        "bwd_fin": flow["bwd_fin"],
        "fwd_rst": flow["fwd_rst"],
        "bwd_rst": flow["bwd_rst"],
        "bwd_psh": flow["bwd_psh"],
        "total_syn": total_syn,
        "total_fin": total_fin,
        "total_rst": total_rst,
        "syn_ratio": round(total_syn/(tp+eps), 4),
        "http_get": flow["http_get"],
        "http_post": flow["http_post"],
        "http_post_ratio": http_post_ratio,
        "unique_dst_ips": len(set(flow["dst_ips"])),
        "payload_entropy": entropy,
        "sql_keyword_count": sqli_feats["sql_keyword_count"],
        "quote_count": sqli_feats["quote_count"],
        "comment_count": sqli_feats["comment_count"],
        "classic_injection": sqli_feats["classic_injection"],
        "hex_encoding_ratio": sqli_feats["hex_encoding_ratio"],
        "time_based_indicators": sqli_feats["time_based_indicators"],
        "is_tls": tls_feats["is_tls"],
        "periodicity_power_ratio": p_ratio,
        "fwd_iat_cv": fwd_iat_cv,
        "bwd_iat_cv": bwd_iat_cv,
        "fwd_len_cv": fwd_len_cv,
        "bwd_len_cv": bwd_len_cv,
        "iat_cv": iat_cv,
        "proto_1": proto_1,
        "proto_2": proto_2,
        "proto_6": proto_6,
        # Non-feature fields used only in overrides / display
        "src_ip": src,
        "dst_ip": dst,
        "src_port": sport,       # FIX-6: exposed so Override 1 can check both ports
        "total_pkts": tp,
        "sqli_hard_flag": sqli_feats["sqli_hard_flag"],
        "http_get_rate": http_get_rate,
    }

# ------------------------------------------------------------
# Temporal Correlation Engine
# ------------------------------------------------------------
class TemporalCorrelationEngine:
    def __init__(self, window_secs=10, alert_threshold=3):
        self.window_secs = window_secs
        self.alert_threshold = alert_threshold
        self._events = defaultdict(list)
        self.total_flows = 0
        self.suppressed = 0
        self.alerted = 0

    def process(self, src_ip, timestamp, predicted_class, confidence=1.0):
        self.total_flows += 1
        if predicted_class == "Normal":
            return "NORMAL", 0
        cutoff = timestamp - self.window_secs
        self._events[src_ip] = [e for e in self._events[src_ip] if e[0] > cutoff]
        self._events[src_ip].append((timestamp, predicted_class))
        count = len(self._events[src_ip])
        if count >= self.alert_threshold:
            self.alerted += 1
            return "ALERT !!", count
        else:
            self.suppressed += 1
            return "LOG_ONLY", count

    def stats(self):
        non_normal = self.alerted + self.suppressed
        return {
            "total_flows": self.total_flows,
            "alerts_raised": self.alerted,
            "suppressed": self.suppressed,
            "suppression_rate": self.suppressed / max(1, non_normal),
        }

    def reset_state(self):
        """Clear all temporal state."""
        self._events.clear()
        self.total_flows = 0
        self.suppressed = 0
        self.alerted = 0

    # Transparent alias so existing callers don't break
    def clear_events(self):
        self.reset_state()

# ------------------------------------------------------------
# Platform helpers
# ------------------------------------------------------------
def check_root():
    if os.name == "nt":
        try:
            import ctypes
            if not ctypes.windll.shell32.IsUserAnAdmin():
                print(f"{RED}[error] Not running as Administrator.{RESET}")
                print("Right-click Command Prompt → 'Run as administrator', then retry.")
                sys.exit(1)
        except Exception:
            pass
    else:
        if os.geteuid() != 0:
            print(f"{RED}[error] Must run as root on Linux/macOS.{RESET}")
            print(f" sudo python3 {sys.argv[0]} {' '.join(sys.argv[1:])}")
            sys.exit(1)

def auto_detect_iface():
    ifaces = get_if_list()
    for iface in ifaces:
        if "mon" in iface.lower():
            return iface
    skip = {"lo", "loopback"}
    for iface in ifaces:
        if iface.lower() in skip:
            continue
        if any(x in iface.lower() for x in ("docker", "virbr", "veth", "br-")):
            continue
        try:
            from scapy.all import get_if_addr
            addr = get_if_addr(iface)
            if addr and addr != "0.0.0.0":
                return iface
        except Exception:
            pass
    return str(scapy_conf.iface)

def list_interfaces():
    print(f"\n{CYAN}Available interfaces:{RESET}")
    ifaces = get_if_list()
    try:
        from scapy.all import get_if_addr
        for iface in ifaces:
            try:
                addr = get_if_addr(iface)
            except Exception:
                addr = "?"
            print(f" {iface:<20}{DIM}{addr}{RESET}")
    except Exception:
        for iface in ifaces:
            print(f" {iface}")
    if os.name == "nt":
        print(f"\n{YELLOW}Windows tip: copy the full \\Device\\NPF_{{...}} string for --iface{RESET}")

# ------------------------------------------------------------
# Live IDS runner
# ------------------------------------------------------------
class LiveIDS:
    def __init__(self, model_path, iface, window_secs=10, alert_threshold=3):
        self.whitelist: set = set()
        print(f"\n{CYAN}[IDS] Loading model bundle from: {model_path}{RESET}")
        self.bundle = joblib.load(model_path)
        self.model = self.bundle["model"]
        self.scaler = self.bundle["scaler"]
        self.features = self.bundle["features"]
        self.features_aug = self.bundle.get("features_aug", self.features)
        self._feat_tuple = tuple(self.features)
        self.label_map = self.bundle["label_map"]
        self.c2_thresh = self.bundle.get("c2_threshold", C2_THRESHOLD)
        self.sql_idx = self.bundle.get("sqli_hard_flag_idx")
        self.iso = self.bundle.get("iso_forest")
        self.use_iso = self.bundle.get("use_iso_forest", False)
        self.conf_thresh = self.bundle.get("confidence_thresholds", CONFIDENCE_THRESHOLDS)

        print(f" Model version    : {self.bundle.get('model_version', 'unknown')}")
        print(f" Features (base)  : {len(self.features)}")
        print(f" Isolation forest : {'enabled' if self.use_iso else 'disabled'}")

        self.iface = iface
        self.engine = TemporalCorrelationEngine(window_secs, alert_threshold)
        self.active_flows = {}
        self.last_timeout_check = time.time()
        self.flow_count = 0
        self.alert_count = 0

        self._agg = defaultdict(lambda: defaultdict(lambda: {
            "first_seen": None, "last_seen": None, "flows": 0, "pkts": 0, "max_conf": 0.0
        }))
        self._ip_alerted = {}
        self._last_agg_flush = time.time()

        try:
            from scapy.all import get_if_list, get_if_addr
            for _iface in get_if_list():
                try:
                    addr = get_if_addr(_iface)
                    if addr and addr != "0.0.0.0":
                        self.whitelist.add(addr)
                except Exception:
                    pass
        except Exception:
            pass

    def reset_state(self):
        """Clear all aggregation, temporal engine, and flow counters."""
        self._agg.clear()
        self._ip_alerted.clear()
        self.engine.reset_state()
        self.active_flows.clear()
        self.flow_count = 0
        self.alert_count = 0
        self._last_agg_flush = time.time()

    def _predict_flow(self, fid, flow, force_syn_flood=False):
        """
        Classify a completed flow.

        Parameters
        ----------
        force_syn_flood : bool
            When True the SYN early-exit path has already determined this is a
            SYN-only flow; skip ML and hard-set pred=1 (SYN_Flood).
        """
        row = finalize_flow(fid, flow)
        src_ip = row["src_ip"]

        # ── FIX-1: SYN early-exit forces class, no ML needed ────────────────
        if force_syn_flood:
            cls_name = "SYN_Flood"
            conf = 1.0
            ts = time.time()
            decision, count = self.engine.process(src_ip, ts, cls_name, conf)
            return {
                "src_ip": src_ip,
                "dst_ip": row["dst_ip"],
                "dst_port": row["dst_port"],
                "predicted_class": cls_name,
                "confidence": conf,
                "conf_passed": True,
                "temporal_count": count,
                "final_decision": decision,
                "duration": row["duration"],
                "total_pkts": row["total_pkts"],
            }

        # ── Normal ML path ───────────────────────────────────────────────────
        X = np.array([[row.get(f, 0.0) for f in self._feat_tuple]], dtype=np.float32)
        X_scaled = self.scaler.transform(X).astype(np.float32)

        if self.use_iso and self.iso is not None:
            iso_score = self.iso.decision_function(X_scaled).reshape(-1, 1)
            X_scaled = np.hstack([X_scaled, iso_score]).astype(np.float32)

        proba = self.model.predict_proba(X_scaled)[0]
        pred = int(np.argmax(proba))
        conf = float(np.max(proba))

        # ── Extract forensic signals once for use in every override ──────────
        dst_port      = row.get("dst_port", 0)
        src_port      = row.get("src_port", 0)           # FIX-6
        all_ports     = {src_port, dst_port}              # FIX-6: bidirectional set
        pkt_rate      = row.get("pkt_rate", 0)
        sql_kw        = row.get("sql_keyword_count", 0)
        sqli_hard     = row.get("sqli_hard_flag", 0)
        syn_ratio     = row.get("syn_ratio", 0)
        total_pkts    = row.get("total_pkts", 0)
        http_get_rate = row.get("http_get_rate", 0)       # FIX-8
        no_sql_evidence = (sql_kw == 0 and sqli_hard == 0)

        # =================================================================
        # CAUSAL OVERRIDES  (applied top-to-bottom; later rules see updated pred)
        # =================================================================

        # Override 1 – Brute Force on management/auth ports
        # FIX-6: check BOTH ports in the flow. flow_key() sorts by (ip, port)
        # lexicographically, so when IPs differ the management port may end up
        # as either src_port or dst_port depending on IP ordering. Checking only
        # dst_port caused silent misses when Kali's high ephemeral port sorted
        # before the management port (e.g. src=55482, dst=23 → canonical key
        # might flip them). No SQL evidence guard: if SQL evidence IS present,
        # let Override 3 (sqli_hard) have the final say below.
        if any(p in BRUTE_FORCE_PORTS for p in all_ports) and pred != 0:
            pred = 4   # Brute_Force

        # Override 2 – HTTP Flood volumetric guard (original)
        # FIX-5: also fires when pred==4 (Brute_Force) to stop false positives.
        # FIX-8: also fires on elevated http_get_rate even without high pkt_rate,
        #        catching slower GET floods that stay under the 50 pkt/s threshold.
        if dst_port in HTTP_PAYLOAD_PORTS and no_sql_evidence:
            if pkt_rate > 50 or http_get_rate > 30:
                pred = 2   # HTTP_Flood

        # AGGRESSIVE VOLUMETRIC GUARD – overrides SQLi hallucination on high-rate port 80
        # If port 80 is involved, rate > 100 pkt/s, model says SQLi (5) but we have
        # no hard SQL evidence (sqli_hard == 0), then it's a volumetric flood.
        if 80 in all_ports and pkt_rate > 100 and pred == 5 and sqli_hard == 0:
            pred = 2   # HTTP_Flood

        # Override 3 – SQLi hard-flag: DPI found real SQL in the payload.
        # Runs AFTER Override 1 so it can restore SQLi even if brute-force port
        # logic demoted it (FIX-4).
        if sqli_hard:
            pred = 5   # SQL_Injection

        # Override 4 – C2 low-rate heuristic
        if (no_sql_evidence and
                proba[3] >= self.c2_thresh and
                total_pkts >= 8 and
                pkt_rate < 10):
            pred = 3   # Command_Control

        # Override 5 – SYN-only disambiguation (belt-and-suspenders)
        if syn_ratio > SYN_ONLY_RATIO_THRESHOLD and pred != 0:
            pred = 1   # SYN_Flood

        # =================================================================
        # Confidence gate
        # =================================================================
        cls_name  = self.label_map[pred]
        threshold = self.conf_thresh.get(cls_name, 0.75)
        conf_pass = (cls_name == "Normal") or (threshold is None) or (conf >= threshold)

        # FIX-4: hard-flag overrides confidence gate on the *final* pred label
        if cls_name == "SQL_Injection" and sqli_hard:
            conf_pass = True

        ts = time.time()
        if conf_pass:
            decision, count = self.engine.process(src_ip, ts, cls_name, conf)
        else:
            decision, count = "SUPPRESSED_LOW_CONF", 0

        return {
            "src_ip": src_ip,
            "dst_ip": row["dst_ip"],
            "dst_port": dst_port,
            "predicted_class": cls_name,
            "confidence": conf,
            "conf_passed": conf_pass,
            "temporal_count": count,
            "final_decision": decision,
            "duration": row["duration"],
            "total_pkts": total_pkts,
        }

    def _flush_timed_out(self):
        now = time.time()
        if now - self.last_timeout_check < TIMEOUT_CHECK_INTERVAL:
            return
        self.last_timeout_check = now
        timed_out = [k for k, v in self.active_flows.items()
                     if v["last_ts"] and (now - v["last_ts"]) > FLOW_TIMEOUT]
        for k in timed_out:
            result = self._predict_flow(k, self.active_flows.pop(k))
            if result:
                self._print_result(result)

    def _is_syn_only_flow(self, flow):
        """Return True if this flow is overwhelmingly SYN with almost no ACKs."""
        total_pkts = flow["fwd_pkts"] + flow["bwd_pkts"]
        if total_pkts < SYN_EARLY_EXIT_PKTS:
            return False
        total_syn = flow["fwd_syn"] + flow["bwd_syn"]
        total_ack = flow["fwd_ack"] + flow["bwd_ack"]
        denom = total_syn + total_ack
        if denom == 0:
            return False
        return (total_syn / denom) >= SYN_ONLY_RATIO_THRESHOLD

    def _packet_callback(self, pkt):
        # FIX-7: reap timed-out flows BEFORE any processing to prevent flow table bloat
        self._flush_timed_out()

        if IP not in pkt:
            return
        key, is_fwd = flow_key(pkt)
        if key is None:
            return
        ts = float(pkt.time)

        # Expire old flow with same 5-tuple
        if key in self.active_flows and (ts - self.active_flows[key]["last_ts"]) > FLOW_TIMEOUT:
            result = self._predict_flow(key, self.active_flows.pop(key))
            if result:
                self._print_result(result)

        if key not in self.active_flows:
            self.active_flows[key] = new_flow()
        update_flow(self.active_flows[key], pkt, is_fwd)

        flow = self.active_flows[key]
        total_pkts = flow["fwd_pkts"] + flow["bwd_pkts"]

        # FIX-1: SYN-only early exit forces SYN_Flood, bypasses ML entirely.
        if TCP in pkt and self._is_syn_only_flow(flow):
            result = self._predict_flow(key, self.active_flows.pop(key),
                                        force_syn_flood=True)
            if result:
                self._print_result(result)

        elif total_pkts >= MIN_PKTS_FOR_CLASSIFICATION:
            result = self._predict_flow(key, self.active_flows.pop(key))
            if result:
                self._print_result(result)

        elif TCP in pkt and (int(pkt[TCP].flags) & 0x05):
            # FIN or RST: immediate close
            result = self._predict_flow(key, self.active_flows.pop(key))
            if result:
                self._print_result(result)

    def _print_result(self, r):
        self.flow_count += 1
        decision = r["final_decision"]
        cls = r["predicted_class"]
        src_ip = r["src_ip"]
        now = time.time()

        if decision == "NORMAL":
            return

        bucket = self._agg[src_ip][cls]
        if bucket["first_seen"] is None:
            bucket["first_seen"] = now
        bucket["last_seen"] = now
        bucket["flows"] += 1
        bucket["pkts"] += r["total_pkts"]
        bucket["max_conf"] = max(bucket["max_conf"], r["confidence"])

        if decision == "ALERT !!":
            self.alert_count += 1
            last_alert = self._ip_alerted.get(src_ip, 0.0)
            if (now - last_alert) >= ALERT_COOLDOWN_SECS:
                self._ip_alerted[src_ip] = now
                color = LABEL_COLORS.get(cls, RESET)
                total_flows = sum(b["flows"] for b in self._agg[src_ip].values())
                ts_str = time.strftime("%H:%M:%S")
                sys.stdout.write("\r\033[K")
                sys.stdout.write(
                    f"{RED}[ALERT]{RESET} {ts_str} "
                    f"src={src_ip:<16} "
                    f"class={color}{cls:<18}{RESET} "
                    f"conf={r['confidence']:.2f} flows={total_flows}\n"
                )
                self._last_agg_flush = 0

        if (now - self._last_agg_flush) >= AGG_FLUSH_SECS:
            self._flush_agg()

    def _flush_agg(self):
        now = time.time()
        self._last_agg_flush = now
        rows = []
        for src_ip, cls_dict in self._agg.items():
            for cls, b in cls_dict.items():
                if b["flows"]:
                    rows.append((src_ip, cls, b["flows"], b["pkts"], b["max_conf"],
                                 now - b["first_seen"]))
        if not rows:
            return
        parts = []
        for src_ip, cls, flows, pkts, max_conf, age in sorted(rows, key=lambda x: -x[2]):
            color = LABEL_COLORS.get(cls, RESET)
            parts.append(f"{color}{src_ip}{RESET} {cls} {flows}flows {age:.0f}s")
        summary = f"{DIM}live{RESET} | " + " | ".join(parts)
        sys.stdout.write(f"\r\033[K{summary}")
        sys.stdout.flush()

    def run(self):
        print(f"\n{CYAN}[IDS] Sniffing on interface: {self.iface}{RESET}")
        print(f"{DIM}Non-normal flows aggregated below. Ctrl+C to stop.{RESET}\n")
        print(f"{'TIME':<10} {'SRC IP':<18} {'CLASS':<20} {'CONF':<6} FLOWS")
        print("-" * 100)
        try:
            sniff(iface=self.iface, prn=self._packet_callback, store=False)
        except KeyboardInterrupt:
            pass
        finally:
            self._flush_agg()
            self._print_summary()

    def _print_summary(self):
        stats = self.engine.stats()
        print("\n" + "-" * 60)
        print(f"{CYAN}Session summary{RESET}")
        print(f" Flows processed  : {stats['total_flows']}")
        print(f" Alerts raised    : {stats['alerts_raised']}")
        print(f" Logged (no alert): {stats['suppressed']}")
        print(f" Suppression rate : {stats['suppression_rate']:.1%}")
        print(f"\n Attacker IPs seen ({len(self._agg)}):")
        for src_ip, cls_dict in sorted(self._agg.items()):
            for cls, b in cls_dict.items():
                if b["flows"]:
                    print(f" {src_ip:<18}{LABEL_COLORS.get(cls,RESET)}{cls}{RESET} - "
                          f"{b['flows']} flows, {b['pkts']} pkts, max conf {b['max_conf']:.2f}")
        print("-" * 60)

# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Live IDS Inference – Exact 59‑feature parity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Kali quick-start:
  sudo apt install -y python3-pip libpcap-dev
  pip3 install scapy xgboost scikit-learn scipy pandas numpy joblib
  sudo python3 ids_live.py --iface eth0

Windows quick-start:
  1. Install Npcap from https://npcap.com (tick 'WinPcap API-compatible mode')
  2. Open Command Prompt as Administrator
  3. python ids_live.py --list
  4. python ids_live.py --model ids_final_model_updated.pkl --iface "\\Device\\NPF_{...}"
        """
    )
    parser.add_argument("--model", default="ids_final_model_updated.pkl",
                        help="Path to the joblib model bundle (.pkl)")
    parser.add_argument("--iface", default=None,
                        help="Network interface (auto-detected if omitted)")
    parser.add_argument("--list", action="store_true",
                        help="List available interfaces and exit")
    parser.add_argument("--window", type=int, default=10,
                        help="Temporal engine window in seconds (default: 10)")
    parser.add_argument("--threshold", type=int, default=3,
                        help="Temporal engine alert threshold (default: 3)")
    args = parser.parse_args()

    if args.list:
        list_interfaces()
        return

    check_root()

    if not os.path.exists(args.model):
        parser.error(f"Model file not found: {args.model}\n"
                     f"Pass the correct path with --model <path>")

    iface = args.iface or auto_detect_iface()
    print(f"{CYAN}[IDS] Interface: {iface}{RESET}" +
          (f"{DIM} (auto-detected){RESET}" if not args.iface else ""))

    ids = LiveIDS(
        model_path=args.model,
        iface=iface,
        window_secs=args.window,
        alert_threshold=args.threshold,
    )
    ids.run()

if __name__ == "__main__":
    main()