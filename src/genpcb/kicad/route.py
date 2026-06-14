"""Routing reward（ground truth）：board → DSN → Freerouting → SES → routed_fraction。

**慢**（Freerouting 每板數十秒），是 Tier-2 ground truth——用於 audit 與訓練 Model A
routability surrogate，**不直接進 GRPO inner loop**（那要 surrogate）。需 Java + Freerouting。
"""

from __future__ import annotations

import glob
import os
import re
import signal
import subprocess


def java_bin() -> str:
    """找 Java 25+ 的 binary。

    Colab 裝 KiCad 會把 `java` 的 alternative 搶回 openjdk-17（class 61），但 Freerouting
    需 Java 25（class 69）。Temurin 25 仍在 /usr/lib/jvm/*25*，直接用其完整路徑，繞過
    `java` 符號連結。可用環境變數 GENPCB_JAVA 覆寫。
    """
    env = os.environ.get("GENPCB_JAVA")
    if env:
        return env
    for p in sorted(glob.glob("/usr/lib/jvm/*25*/bin/java")):
        return p
    return "java"

from genpcb.data.procedural import Board
from genpcb.kicad.dsn import board_to_dsn
from genpcb.kicad.viz import parse_ses_geometry


def parse_freerouting_log(stdout: str) -> dict:
    """從完整 Freerouting log 抽 routed_fraction（'started with N ... (M unrouted)'）。"""
    start = re.search(r"started with (\d+) unrouted", stdout)
    finals = re.findall(r"\((\d+) unrouted\)", stdout)
    s = int(start.group(1)) if start else None
    f = int(finals[-1]) if finals else None
    rf = ((s - f) / s) if (s and f is not None and s > 0) else None
    return {"unrouted_start": s, "unrouted_final": f, "routed_fraction": rf}


def _last_unrouted(stdout: str) -> int | None:
    """最後一個 '(N unrouted)'——完整或被 timeout 砍斷的 log 都適用。"""
    finals = re.findall(r"\((\d+) unrouted\)", stdout)
    return int(finals[-1]) if finals else None


def total_connections(board: Board) -> int:
    """需佈線的連線總數（每 net 的 spanning-tree 邊 = pins-1）≈ Freerouting 起始 unrouted。"""
    return sum(max(0, len(n.pins) - 1) for n in board.nets)


def run_freerouting(dsn_path: str, ses_path: str, jar: str = "/content/freerouting.jar",
                    max_passes: int = 5, timeout: int = 90, headless: bool = True) -> tuple[bool, str, str]:
    """跑 Freerouting。回傳 (是否產出 ses, stdout, stderr, 是否 timeout)。

    旗標 -de/-do/-mp 為 Freerouting core；若版本不同，依其 --help 調整本函式。
    headless=True 用 xvfb-run（Colab 無顯示）。timeout 也收回部分 stdout（供算部分進度）。
    """
    if os.path.exists(ses_path):
        os.remove(ses_path)
    base = [java_bin(), "-jar", jar, "-de", dsn_path, "-do", ses_path, "-mp", str(max_passes)]
    cmd = (["xvfb-run", "-a"] + base) if headless else base
    # start_new_session → 可整個 process group 砍掉（避免 timeout 後 java 殘留累積）
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = proc.communicate()                 # 收回 timeout 前已產生的 stdout
        timed_out = True
    return os.path.exists(ses_path), out, err, timed_out


def routing_reward(board: Board, jar: str = "/content/freerouting.jar", workdir: str = "/tmp",
                   max_passes: int = 5, fail_reward: float = -1.0) -> dict:
    """單板 routing ground truth。reward = routed_fraction（v0）。

    routed_fraction = (總連線數 − 最後 unrouted) / 總連線數。**timeout 不丟資料**：用被砍斷
    log 裡最後一個 '(N unrouted)' → 難佈的板得低分（有用訊號），不是 None。
    只有連 pass 1 都沒到（無任何 '(N unrouted)'）才算真失敗 → None。
    """
    dpath = os.path.join(workdir, "rr.dsn")
    spath = os.path.join(workdir, "rr.ses")
    with open(dpath, "w") as f:
        f.write(board_to_dsn(board))
    _ses_ok, out, err, timed_out = run_freerouting(dpath, spath, jar=jar, max_passes=max_passes)

    total = total_connections(board) or 1
    unrouted = _last_unrouted(out)
    if unrouted is None:
        if timed_out:
            # Freerouting 有跑但連 pass 1 都沒完成 → 極難佈 → rf=0（有效低分，非排除）
            return {"ok": True, "routed_fraction": 0.0, "reward": 0.0, "total": total,
                    "unrouted": total, "timed_out": True, "n_wires": 0, "n_vias": 0}
        # 非 timeout 卻無任何進度 → env/DSN 錯 → None（排除，resume 重試）
        return {"ok": False, "routed_fraction": None, "reward": fail_reward,
                "timed_out": False, "stdout_tail": out[-400:], "stderr": err[-400:]}
    rf = max(0.0, (total - unrouted) / total)
    g = parse_ses_geometry(open(spath).read()) if os.path.exists(spath) else {"wires": [], "vias": []}
    return {"ok": True, "routed_fraction": rf, "reward": rf, "total": total, "unrouted": unrouted,
            "timed_out": timed_out, "n_wires": len(g["wires"]), "n_vias": len(g["vias"])}
