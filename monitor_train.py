"""Training monitor: pushes status to WeChat (ServerChan) and/or iPhone (Bark).

Usage:
    export SCT_KEY=SCTxxxxxxxx      # ServerChan Turbo SendKey (WeChat)
    export BARK_KEY=xxxxxxxx        # optional, Bark device key (iPhone)
    (or put the keys in ~/.config/sct_key / ~/.config/bark_key, so tmux/@reboot launches get them too)
    nohup python monitor_train.py > monitor.log 2>&1 &

Sends a summary every REPORT_EVERY seconds, and an immediate alert when the
training process dies, loss becomes NaN, or the log stops updating.
"""
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request

LOG = os.environ.get("TRAIN_LOG", "/home/mlp4/data/ZHOU/SSR/training_v3.log")
PROC_PATTERN = os.environ.get("TRAIN_PROC", "Train_v3.py")
TOTAL_EPOCHS = int(os.environ.get("TOTAL_EPOCHS", 300))
REPORT_EVERY = int(os.environ.get("REPORT_EVERY", 2 * 3600))
CHECK_EVERY = int(os.environ.get("CHECK_EVERY", 300))
STALL_SECONDS = int(os.environ.get("STALL_SECONDS", 900))

EPOCH_RE = re.compile(r"epoch =\s+(\d+) , loss = ([\d.]+) , Avg PSNR = ([\d.]+) ,time = ([\d.]+) s")


def _file_key(name):
    try:
        with open(os.path.expanduser(f"~/.config/{name}")) as f:
            return f.read().strip()
    except OSError:
        return ""


def push(title, body):
    print(f"[push] {title}\n{body}", flush=True)
    sct, bark = os.environ.get("SCT_KEY") or _file_key("sct_key"), os.environ.get("BARK_KEY") or _file_key("bark_key")
    if not (sct or bark):
        print("push skipped: no SCT_KEY/BARK_KEY (env or ~/.config/sct_key, ~/.config/bark_key)", flush=True)
        return
    try:
        if sct:
            data = urllib.parse.urlencode({"title": title, "desp": body}).encode()
            urllib.request.urlopen(f"https://sctapi.ftqq.com/{sct}.send", data, timeout=15)
        if bark:
            url = "https://api.day.app/%s/%s/%s" % (
                bark, urllib.parse.quote(title), urllib.parse.quote(body))
            urllib.request.urlopen(url, timeout=15)
    except Exception as e:
        print("push failed:", e, flush=True)


def alive():
    out = subprocess.run(["pgrep", "-f", f"python.*{PROC_PATTERN}"], capture_output=True, text=True)
    return bool(out.stdout.strip())


def read_tail(n=300_000):
    with open(LOG, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - n))
        return f.read().decode(errors="ignore")


def epochs():
    with open(LOG, errors="ignore") as f:
        return [tuple(map(float, m.groups())) for m in EPOCH_RE.finditer(f.read())]


def summary():
    eps = epochs()
    if not eps:
        return "暂无已完成的 epoch"
    e, loss, psnr, t = eps[-1]
    prev = eps[-25][2] if len(eps) > 25 else eps[0][2]
    avg_t = sum(x[3] for x in eps[-10:]) / len(eps[-10:])
    left_h = (TOTAL_EPOCHS - e) * avg_t / 3600
    return (f"epoch {int(e)}/{TOTAL_EPOCHS}\n\nloss {loss:.5f}\n\nPSNR {psnr:.2f} dB "
            f"(近 24 epoch {psnr - prev:+.2f})\n\n每 epoch {avg_t:.0f}s，预计剩余 {left_h:.1f} 小时")


def main():
    last_report = 0
    alerted = set()
    push("监控已启动", summary())
    while True:
        now = time.time()
        problems = []
        if not alive():
            problems.append("died")
        else:
            alerted.discard("died")
        if "nan" in read_tail().lower().replace("nanosec", ""):
            problems.append("nan")
        if now - os.path.getmtime(LOG) > STALL_SECONDS and alive():
            problems.append("stall")
        else:
            alerted.discard("stall")
        for p in problems:
            if p not in alerted:
                alerted.add(p)
                msg = {"died": "训练进程已退出", "nan": "日志中出现 NaN", "stall": "日志长时间无更新"}[p]
                push("训练告警：" + msg, summary())
        if now - last_report >= REPORT_EVERY:
            last_report = now
            push("训练进度", summary())
        if "died" in problems and TOTAL_EPOCHS and epochs() and epochs()[-1][0] >= TOTAL_EPOCHS:
            push("训练完成", summary())
            break
        time.sleep(CHECK_EVERY)


if __name__ == "__main__":
    main()
