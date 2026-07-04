"""
ids_gui.py  —  Tkinter GUI front-end for ids_live.py
Runs the IDS in a background thread; GUI polls a queue every 250 ms.

Requirements (same as ids_live.py, tkinter is built-in):
    pip install scapy xgboost scikit-learn scipy pandas numpy joblib

Run (must be admin/root the same as ids_live.py):
    Windows:   python ids_gui.py
    Linux/Mac: sudo python3 ids_gui.py
"""

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
import os

# ── Colour palette (works on both light and dark OS themes) ──────────────────
BG          = "#1e1e2e"   # window background
BG2         = "#27273a"   # panel background
BG3         = "#2e2e42"   # row / entry background
FG          = "#cdd6f4"   # normal text
FG2         = "#a6adc8"   # muted text
ALERT_BG    = "#3b1f1f"
ALERT_FG    = "#f38ba8"
ALERT_ACC   = "#f38ba8"   # red accent
LOG_BG      = "#2e2a1a"
LOG_FG      = "#f9e2af"
LOG_ACC     = "#f9e2af"   # amber accent
NORMAL_BG   = "#1a2e1f"
NORMAL_FG   = "#a6e3a1"
NORMAL_ACC  = "#a6e3a1"   # green accent
HEADER_BG   = "#181825"
ACCENT      = "#89b4fa"   # blue for headers / highlights

CLASS_COLORS = {
    "SYN_Flood":       "#f38ba8",
    "HTTP_Flood":      "#fab387",
    "Command_Control": "#cba6f7",
    "Brute_Force":     "#f38ba8",
    "SQL_Injection":   "#89b4fa",
    "Normal":          "#a6e3a1",
}

MAX_FEED_ROWS = 200   # keep feed from growing forever


# ─────────────────────────────────────────────────────────────────────────────
#  Queue bridge — ids_live._print_result posts here instead of stdout
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  Interface discovery
# ─────────────────────────────────────────────────────────────────────────────

def get_interfaces():
    """Returns list of (friendly_name, ip_address, raw_iface_id) tuples."""
    results = []
    try:
        from scapy.all import IFACES, get_if_list, get_if_addr
        if hasattr(IFACES, 'data'):
            seen = set()
            for iface_obj in IFACES.data.values():
                raw = getattr(iface_obj, 'name', str(iface_obj))
                if raw in seen:
                    continue
                seen.add(raw)
                friendly = getattr(iface_obj, 'description',
                           getattr(iface_obj, 'network_name', raw))
                ip = getattr(iface_obj, 'ip', '')
                if not ip:
                    try:
                        ip = get_if_addr(raw)
                    except Exception:
                        ip = ''
                if ip in ('0.0.0.0', None):
                    ip = ''
                results.append((friendly or raw, ip or '', raw))
            if results:
                return results
        for raw in get_if_list():
            try:
                ip = get_if_addr(raw)
            except Exception:
                ip = ''
            if ip in ('0.0.0.0', None):
                ip = ''
            results.append((raw, ip or '', raw))
    except Exception:
        pass
    return results or [('(no interfaces found)', '', '')]


event_queue: queue.Queue = queue.Queue()


def _patch_ids(ids_instance):
    """
    Monkey-patch LiveIDS._print_result so that every classified flow is
    also posted to event_queue.  The original terminal output still works.
    """
    original = ids_instance._print_result

    def patched(r):
        original(r)                 # keep existing terminal behaviour
        event_queue.put(dict(r))    # also send to GUI

    ids_instance._print_result = patched


# ─────────────────────────────────────────────────────────────────────────────
#  Background sniff thread
# ─────────────────────────────────────────────────────────────────────────────

_sniff_thread = None
_stop_flag    = threading.Event()


def start_sniffing(model_path, iface, window_secs, alert_threshold, status_cb):
    global _sniff_thread, _stop_flag

    _stop_flag.clear()

    def run():
        try:
            # Import here so errors surface in the GUI, not at startup
            import ids_live as ids_mod
            ids_mod.check_root()

            status_cb("Loading model…")
            ids_obj = ids_mod.LiveIDS(
                model_path      = model_path,
                iface           = iface,
                window_secs     = window_secs,
                alert_threshold = alert_threshold,
            )
            _patch_ids(ids_obj)
            status_cb(f"Sniffing on {iface}")

            from scapy.all import sniff as scapy_sniff
            scapy_sniff(
                iface   = iface,
                prn     = ids_obj._packet_callback,
                store   = False,
                stop_filter = lambda _: _stop_flag.is_set(),
            )
            status_cb("Stopped")
        except Exception as exc:
            status_cb(f"Error: {exc}")
            event_queue.put({"_error": str(exc)})

    _sniff_thread = threading.Thread(target=run, daemon=True)
    _sniff_thread.start()


def stop_sniffing():
    _stop_flag.set()


# ─────────────────────────────────────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────────────────────────────────────

class IDSApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IDS Live Monitor")
        self.geometry("1100x700")
        self.minsize(900, 580)
        self.configure(bg=BG)

        # counters
        self._total   = 0
        self._alerts  = 0
        self._logs    = 0
        self._normal  = 0

        self._build_ui()
        self._poll()   # start queue polling

    # ── Build ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_topbar()
        self._build_stats()
        self._build_main()
        self._build_statusbar()

    def _build_topbar(self):
        bar = tk.Frame(self, bg=HEADER_BG, height=52)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        tk.Label(bar, text="● IDS Live Monitor", bg=HEADER_BG, fg=ACCENT,
                 font=("Courier", 14, "bold")).pack(side="left", padx=16, pady=14)

        # Controls on the right
        ctrl = tk.Frame(bar, bg=HEADER_BG)
        ctrl.pack(side="right", padx=12)

        # Model picker
        tk.Label(ctrl, text="Model:", bg=HEADER_BG, fg=FG2,
                 font=("Courier", 10)).grid(row=0, column=0, padx=(0,4))
        self._model_var = tk.StringVar(value="ids_final_model_updated.pkl")
        tk.Entry(ctrl, textvariable=self._model_var, width=28,
                 bg=BG3, fg=FG, insertbackground=FG, relief="flat",
                 font=("Courier", 10)).grid(row=0, column=1, padx=(0,4))
        tk.Button(ctrl, text="…", bg=BG3, fg=FG, relief="flat",
                  font=("Courier", 10), cursor="hand2",
                  command=self._browse_model).grid(row=0, column=2, padx=(0,10))

        # Interface picker - button opens popup table
        self._iface_raw = ''
        self._iface_label_var = tk.StringVar(value='(none selected)')
        tk.Button(ctrl, text='Interface...', bg=BG3, fg=FG, relief='flat',
                  font=('Courier', 10), cursor='hand2',
                  command=self._pick_iface).grid(row=0, column=3, padx=(0,4))
        tk.Label(ctrl, textvariable=self._iface_label_var,
                 bg=HEADER_BG, fg=ACCENT, font=('Courier', 10),
                 width=24, anchor='w').grid(row=0, column=4, padx=(0,10))

        # Start / Stop
        self._start_btn = tk.Button(ctrl, text="▶  Start", bg="#1e3a2a", fg=NORMAL_ACC,
                                    relief="flat", font=("Courier", 10, "bold"),
                                    cursor="hand2", padx=10,
                                    command=self._start)
        self._start_btn.grid(row=0, column=5, padx=(0,4))

        self._stop_btn = tk.Button(ctrl, text="■  Stop", bg="#3b1f1f", fg=ALERT_FG,
                                   relief="flat", font=("Courier", 10, "bold"),
                                   cursor="hand2", padx=10, state="disabled",
                                   command=self._stop)
        self._stop_btn.grid(row=0, column=6, padx=(0,4))

        tk.Button(ctrl, text="Clear", bg=BG3, fg=FG2, relief="flat",
                  font=("Courier", 10), cursor="hand2",
                  command=self._clear).grid(row=0, column=7)

    def _build_stats(self):
        bar = tk.Frame(self, bg=BG, pady=8)
        bar.pack(fill="x", padx=12)

        cards = [
            ("Flows",   "_total_lbl",  FG,         "0"),
            ("Alerts",  "_alert_lbl",  ALERT_ACC,  "0"),
            ("Log only","_log_lbl",    LOG_ACC,    "0"),
            ("Normal",  "_norm_lbl",   NORMAL_ACC, "0"),
        ]
        for i, (label, attr, color, val) in enumerate(cards):
            f = tk.Frame(bar, bg=BG2, padx=18, pady=8)
            f.grid(row=0, column=i, padx=6, sticky="ew")
            bar.columnconfigure(i, weight=1)
            tk.Label(f, text=label, bg=BG2, fg=FG2,
                     font=("Courier", 10)).pack(anchor="w")
            lbl = tk.Label(f, text=val, bg=BG2, fg=color,
                           font=("Courier", 20, "bold"))
            lbl.pack(anchor="w")
            setattr(self, attr, lbl)

    def _build_main(self):
        pane = tk.PanedWindow(self, orient="horizontal", bg=BG,
                              sashwidth=4, sashrelief="flat")
        pane.pack(fill="both", expand=True, padx=12, pady=(0,4))

        # ── Left: live feed ──────────────────────────────────────────────────
        left = tk.Frame(pane, bg=BG)
        pane.add(left, minsize=500)

        tk.Label(left, text="Live feed", bg=BG, fg=ACCENT,
                 font=("Courier", 11, "bold")).pack(anchor="w", pady=(4,4))

        cols = ("time","decision","src_ip","class","conf","pkts","port")
        self._feed = ttk.Treeview(left, columns=cols, show="headings",
                                  height=20, selectmode="browse")

        heads = [("time","Time",70), ("decision","Decision",90),
                 ("src_ip","Src IP",120), ("class","Class",120),
                 ("conf","Conf",50), ("pkts","Pkts",50), ("port","Port",50)]
        for cid, text, w in heads:
            self._feed.heading(cid, text=text)
            self._feed.column(cid, width=w, anchor="w", stretch=False)

        # Row tags
        self._feed.tag_configure("ALERT",  background=ALERT_BG,  foreground=ALERT_FG)
        self._feed.tag_configure("LOG",    background=LOG_BG,    foreground=LOG_FG)
        self._feed.tag_configure("NORMAL", background=NORMAL_BG, foreground=NORMAL_FG)

        # Style the treeview
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Treeview",
                        background=BG2, foreground=FG,
                        fieldbackground=BG2, rowheight=22,
                        font=("Courier", 10))
        style.configure("Treeview.Heading",
                        background=HEADER_BG, foreground=ACCENT,
                        font=("Courier", 10, "bold"), relief="flat")
        style.map("Treeview", background=[("selected", BG3)])

        vsb = ttk.Scrollbar(left, orient="vertical", command=self._feed.yview)
        self._feed.configure(yscrollcommand=vsb.set)
        self._feed.pack(side="left", fill="both", expand=True)
        vsb.pack(side="left", fill="y")

        # ── Right: attacker summary ──────────────────────────────────────────
        right = tk.Frame(pane, bg=BG)
        pane.add(right, minsize=220)

        tk.Label(right, text="Attacker summary", bg=BG, fg=ACCENT,
                 font=("Courier", 11, "bold")).pack(anchor="w", pady=(4,4))

        cols2 = ("src_ip","class","flows","max_conf")
        self._summary = ttk.Treeview(right, columns=cols2, show="headings",
                                     height=20, selectmode="browse")
        self._summary.heading("src_ip",   text="Src IP")
        self._summary.heading("class",    text="Class")
        self._summary.heading("flows",    text="Flows")
        self._summary.heading("max_conf", text="MaxConf")
        self._summary.column("src_ip",   width=120, anchor="w", stretch=True)
        self._summary.column("class",    width=110, anchor="w", stretch=True)
        self._summary.column("flows",    width=55,  anchor="e", stretch=False)
        self._summary.column("max_conf", width=65,  anchor="e", stretch=False)
        self._summary.tag_configure("ALERT", background=ALERT_BG, foreground=ALERT_FG)
        self._summary.tag_configure("LOG",   background=LOG_BG,   foreground=LOG_FG)

        vsb2 = ttk.Scrollbar(right, orient="vertical", command=self._summary.yview)
        self._summary.configure(yscrollcommand=vsb2.set)
        self._summary.pack(side="left", fill="both", expand=True)
        vsb2.pack(side="left", fill="y")

        # internal summary data: {(src_ip, cls): {"flows":0, "max_conf":0.0, "decision":""}}
        self._sumdata = {}

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=HEADER_BG, height=24)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self._status_var = tk.StringVar(value="Ready — press ▶ Start")
        tk.Label(bar, textvariable=self._status_var, bg=HEADER_BG, fg=FG2,
                 font=("Courier", 10), anchor="w").pack(side="left", padx=10)

    # ── Actions ──────────────────────────────────────────────────────────────

    def _pick_iface(self):
        ifaces = get_interfaces()
        popup = tk.Toplevel(self)
        popup.title('Select network interface')
        popup.configure(bg=BG)
        popup.geometry('700x280')
        popup.resizable(True, True)
        popup.grab_set()
        tk.Label(popup, text='Double-click or select and press Choose',
                 bg=BG, fg=FG2, font=('Courier', 10)
                 ).pack(anchor='w', padx=12, pady=(10, 4))
        frame = tk.Frame(popup, bg=BG)
        frame.pack(fill='both', expand=True, padx=12, pady=(0, 4))
        cols = ('friendly', 'ip', 'raw')
        tree = ttk.Treeview(frame, columns=cols, show='headings',
                            height=8, selectmode='browse')
        tree.heading('friendly', text='Adapter name')
        tree.heading('ip',       text='IP address')
        tree.heading('raw',      text='Interface ID')
        tree.column('friendly', width=200, anchor='w', stretch=True)
        tree.column('ip',       width=120, anchor='w', stretch=False)
        tree.column('raw',      width=330, anchor='w', stretch=True)
        vsb = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side='left', fill='both', expand=True)
        vsb.pack(side='left', fill='y')
        for friendly, ip, raw in ifaces:
            tree.insert('', 'end', values=(friendly, ip or '--', raw))
        def on_select(event=None):
            sel = tree.selection()
            if not sel:
                return
            vals = tree.item(sel[0])['values']
            friendly, ip, raw = vals[0], vals[1], vals[2]
            self._iface_raw = str(raw)
            display = str(friendly)
            if ip and ip not in ('--', ''):
                display = '{} ({})'.format(friendly, ip)
            self._iface_label_var.set(display[:26])
            popup.destroy()
        tree.bind('<Double-1>', on_select)
        tk.Button(popup, text='Choose', bg='#1e3a2a', fg=NORMAL_ACC,
                  relief='flat', font=('Courier', 10, 'bold'),
                  cursor='hand2', command=on_select).pack(pady=(2, 10))

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select model bundle",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if path:
            self._model_var.set(path)

    def _start(self):
        model = self._model_var.get().strip()
        iface = self._iface_raw.strip()
        if not model:
            messagebox.showerror("Missing model", "Please specify a model .pkl path.")
            return
        if not iface:
            messagebox.showerror("Missing interface", "Please select a network interface.")
            return
        if not os.path.exists(model):
            messagebox.showerror("Not found", f"Model file not found:\n{model}")
            return

        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._set_status(f"Starting on {iface}…")

        start_sniffing(
            model_path      = model,
            iface           = iface,
            window_secs     = 10,
            alert_threshold = 3,
            status_cb       = self._set_status,
        )

    def _stop(self):
        stop_sniffing()
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        self._set_status("Stopping…")

    def _clear(self):
        for item in self._feed.get_children():
            self._feed.delete(item)
        for item in self._summary.get_children():
            self._summary.delete(item)
        self._sumdata.clear()
        self._total = self._alerts = self._logs = self._normal = 0
        self._update_stats()

    def _set_status(self, msg):
        # safe to call from any thread
        self.after(0, lambda: self._status_var.set(msg))

    # ── Queue polling (runs on main thread via after()) ───────────────────────

    def _poll(self):
        try:
            while True:
                event = event_queue.get_nowait()
                if "_error" in event:
                    messagebox.showerror("IDS Error", event["_error"])
                else:
                    self._handle_event(event)
        except queue.Empty:
            pass
        self.after(250, self._poll)

    def _handle_event(self, r):
        decision = r.get("final_decision", "")
        cls      = r.get("predicted_class", "")
        src_ip   = r.get("src_ip", "")
        conf     = r.get("confidence", 0.0)
        pkts     = r.get("total_pkts", 0)
        port     = r.get("dst_port", 0)
        ts       = time.strftime("%H:%M:%S")

        # Skip pure normals from filling the feed
        if decision == "NORMAL":
            self._normal += 1
            self._total  += 1
            self._update_stats()
            return

        # Decide row tag
        if decision == "ALERT !!":
            tag = "ALERT"
            self._alerts += 1
        elif decision == "LOG_ONLY":
            tag = "LOG"
            self._logs += 1
        else:
            tag = "LOG"   # SUPPRESSED_LOW_CONF etc.
        self._total += 1
        self._update_stats()

        # ── Feed row ─────────────────────────────────────────────────────────
        display_decision = "ALERT !!" if decision == "ALERT !!" else "log only"
        self._feed.insert("", 0, values=(
            ts,
            display_decision,
            src_ip,
            cls,
            f"{conf:.2f}",
            pkts,
            port,
        ), tags=(tag,))

        # Trim feed
        children = self._feed.get_children()
        if len(children) > MAX_FEED_ROWS:
            self._feed.delete(children[-1])

        # ── Attacker summary ─────────────────────────────────────────────────
        key = (src_ip, cls)
        if key not in self._sumdata:
            self._sumdata[key] = {"flows": 0, "max_conf": 0.0, "decision": tag}

        self._sumdata[key]["flows"]    += 1
        self._sumdata[key]["max_conf"]  = max(self._sumdata[key]["max_conf"], conf)
        if tag == "ALERT":
            self._sumdata[key]["decision"] = "ALERT"

        self._refresh_summary()

    def _refresh_summary(self):
        # Rebuild summary treeview sorted by flows descending
        for item in self._summary.get_children():
            self._summary.delete(item)

        sorted_rows = sorted(self._sumdata.items(),
                             key=lambda x: x[1]["flows"], reverse=True)
        for (src_ip, cls), d in sorted_rows:
            tag = d["decision"]
            self._summary.insert("", "end", values=(
                src_ip,
                cls,
                d["flows"],
                f"{d['max_conf']:.2f}",
            ), tags=(tag,))

    def _update_stats(self):
        self._total_lbl.config(text=str(self._total))
        self._alert_lbl.config(text=str(self._alerts))
        self._log_lbl.config(text=str(self._logs))
        self._norm_lbl.config(text=str(self._normal))


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ids_live.py must be in the same folder (or on PYTHONPATH)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    app = IDSApp()
    app.mainloop()
