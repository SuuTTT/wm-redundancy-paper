#!/usr/bin/env python3
"""Live kanban for the SUFFICE 4-box campaign. Polls each GPU box over ssh, renders index.html.
Serve with `python3 -m http.server` in the same dir + cloudflared tunnel."""
import subprocess, time, html, os
KEY = os.path.expanduser("~/.ssh/vastai_id_ed25519")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "board")
os.makedirs(OUT, exist_ok=True)
TARGET = 1500000
BOXES = [
    dict(id="box1", gpu="4×3060", host="124.60.192.12", port=52914, role="TD-MPC2 · DMC front-4 · seeds 0,1"),
    dict(id="box3", gpu="4×4070", host="188.250.227.38", port=28852, role="TD-MPC2 · DMC back-4 · seeds 0,1"),
    dict(id="box2", gpu="4×5060", host="154.12.38.116", port=50649, role="TD-MPC2 · DMC all-8 · seed 2 (n=3)"),
    dict(id="box5", gpu="4×3060", host="156.238.224.242", port=62366, role="Multi-WM · manipulation"),
]
POLL = ("for f in /root/helios-rl/exp/tdmpc_glass/*_vsg_*/seed_*.csv; do [ -f \"$f\" ] || continue; "
        "d=$(basename $(dirname $f)); s=$(tail -1 \"$f\" 2>/dev/null | cut -d, -f1); echo \"$d|$s\"; done; "
        "echo GPU $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr '\\n' ' ')")

def poll(b):
    try:
        r = subprocess.run(["ssh","-i",KEY,"-o","StrictHostKeyChecking=no","-o","ConnectTimeout=15",
                            "-p",str(b["port"]),f"root@{b['host']}",POLL],
                           capture_output=True, text=True, timeout=45)
        out = r.stdout
    except Exception as e:
        return None, f"unreachable ({str(e)[:30]})"
    runs, gpu = {}, ""
    for ln in out.splitlines():
        if ln.startswith("GPU "): gpu = ln[4:].strip(); continue
        if "|" not in ln: continue
        d, s = ln.rsplit("|", 1)
        try: runs[d] = int(s)
        except: pass
    return {"runs": runs, "gpu": gpu}, None

def parse(dname):
    # <Task>_vsg_<Task>_<full|strip>_s<seed>
    try:
        left, right = dname.split("_vsg_", 1)
        arm = "full" if "_full_s" in right else ("strip" if "_strip_s" in right else "?")
        seed = right.rsplit("_s", 1)[1]
        return left, arm, seed
    except Exception:
        return dname, "?", "?"

def render(data, stamp):
    done = running = 0
    box_html = []
    for b in BOXES:
        d = data[b["id"]]
        cards = []
        if d["state"] is None:
            for name, step in sorted(d["data"]["runs"].items()):
                task, arm, seed = parse(name)
                pct = min(100, int(100*step/TARGET))
                fin = step >= TARGET
                if fin: done += 1
                else: running += 1
                color = "#2f6b45" if fin else ("#2e6e8e" if arm=="full" else "#b23a2e")
                cards.append(
                    f'<div class="card"><div class="cardtop"><b>{html.escape(task)}</b>'
                    f'<span class="arm {arm}">{arm}·s{seed}</span></div>'
                    f'<div class="bar"><i style="width:{pct}%;background:{color}"></i></div>'
                    f'<div class="sub">{step:,} / {TARGET:,} · {"done ✓" if fin else str(pct)+"%"}</div></div>')
            status = f'<span class="ok">GPU {html.escape(d["data"]["gpu"])}</span>' if d["data"]["gpu"] else ""
            body = "".join(cards) or '<div class="empty">no runs yet — warming up</div>'
        else:
            status = f'<span class="err">{html.escape(d["state"])}</span>'
            body = '<div class="empty">—</div>'
        box_html.append(
            f'<section class="box"><div class="bhead"><div><h2>{b["id"]} · {b["gpu"]}</h2>'
            f'<div class="role">{html.escape(b["role"])}</div></div>{status}</div>'
            f'<div class="cards">{body}</div></section>')
    total = done + running
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="30">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SUFFICE — live campaign</title>
<style>
:root{{--paper:#faf9f6;--ink:#16191f;--muted:#5b6470;--blue:#2e6e8e;--rule:#e3e1d9}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);
font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;font-size:15px}}
.wrap{{max-width:1100px;margin:0 auto;padding:26px 20px 70px}}
h1{{font-family:Charter,Georgia,serif;font-size:26px;margin:0 0 2px}}
.meta{{color:var(--muted);font-size:13px;margin-bottom:18px}}
.summary{{display:flex;gap:10px;margin:0 0 22px;flex-wrap:wrap}}
.pill{{background:#fff;border:1px solid var(--rule);border-radius:10px;padding:10px 16px;font-variant-numeric:tabular-nums}}
.pill b{{font-size:22px;display:block;font-family:Charter,Georgia,serif}}
.pill.g b{{color:#2f6b45}} .pill.b b{{color:var(--blue)}}
.box{{background:#fff;border:1px solid var(--rule);border-radius:12px;padding:14px 16px;margin-bottom:16px}}
.bhead{{display:flex;justify-content:space-between;align-items:flex-start;border-bottom:1px solid var(--rule);padding-bottom:9px;margin-bottom:11px}}
.bhead h2{{font-size:16px;margin:0}} .role{{color:var(--muted);font-size:12.5px;margin-top:2px}}
.ok{{color:#2f6b45;font-size:12px;font-weight:600}} .err{{color:#b23a2e;font-size:12px;font-weight:600}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(215px,1fr));gap:10px}}
.card{{border:1px solid var(--rule);border-radius:9px;padding:9px 11px;background:var(--paper)}}
.cardtop{{display:flex;justify-content:space-between;align-items:baseline;font-size:13.5px}}
.arm{{font-size:10.5px;font-weight:700;padding:1px 6px;border-radius:10px;color:#fff}}
.arm.full{{background:#2e6e8e}} .arm.strip{{background:#b23a2e}} .arm\\?{{background:#9a9a8c}}
.bar{{height:7px;background:#e7e5de;border-radius:5px;overflow:hidden;margin:7px 0 5px}}
.bar i{{display:block;height:100%}}
.sub{{font-size:11.5px;color:var(--muted);font-variant-numeric:tabular-nums}}
.empty{{color:#9aa1ab;font-size:13px;padding:8px 2px}}
</style></head><body><div class="wrap">
<h1>SUFFICE — live campaign</h1>
<div class="meta">TD-MPC2 component × world-model study · target {TARGET:,} steps/run · updated {stamp} UTC · auto-refresh 30s</div>
<div class="summary">
<div class="pill g"><b>{done}</b>runs done</div>
<div class="pill b"><b>{running}</b>running</div>
<div class="pill"><b>{total}</b>tracked</div>
</div>
{''.join(box_html)}
</div></body></html>"""
    with open(os.path.join(OUT, "index.html"), "w") as f:
        f.write(page)

if __name__ == "__main__":
    while True:
        data = {}
        for b in BOXES:
            d, err = poll(b)
            data[b["id"]] = {"data": d, "state": err}
        render(data, time.strftime("%Y-%m-%d %H:%M:%S"))
        time.sleep(120)
