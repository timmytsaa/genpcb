"""Routing reward（ground truth）：board → DSN → Freerouting → SES → routed_fraction。

**慢**（Freerouting 每板數十秒），是 Tier-2 ground truth——用於 audit 與訓練 Model A
routability surrogate，**不直接進 GRPO inner loop**（那要 surrogate）。需 Java + Freerouting。
"""

from __future__ import annotations

import os
import subprocess

from genpcb.data.procedural import Board
from genpcb.kicad.dsn import board_to_dsn, parse_ses_routed_fraction


def run_freerouting(dsn_path: str, ses_path: str, jar: str = "/content/freerouting.jar",
                    max_passes: int = 5, timeout: int = 900, headless: bool = True) -> tuple[bool, str, str]:
    """跑 Freerouting。回傳 (是否產出 ses, stdout, stderr)。

    旗標 -de/-do/-mp 為 Freerouting core；若版本不同，依其 --help 調整本函式。
    headless=True 用 xvfb-run（Colab 無顯示）。
    """
    if os.path.exists(ses_path):
        os.remove(ses_path)
    base = ["java", "-jar", jar, "-de", dsn_path, "-do", ses_path, "-mp", str(max_passes)]
    cmd = (["xvfb-run", "-a"] + base) if headless else base
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return os.path.exists(ses_path), r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"timeout {timeout}s"


def routing_reward(board: Board, jar: str = "/content/freerouting.jar", workdir: str = "/tmp",
                   max_passes: int = 5, fail_reward: float = -1.0) -> dict:
    """單板 routing ground truth。reward = routed_fraction（v0）；佈線失敗 → fail_reward。"""
    dsn = board_to_dsn(board)
    dpath = os.path.join(workdir, "rr.dsn")
    spath = os.path.join(workdir, "rr.ses")
    with open(dpath, "w") as f:
        f.write(dsn)
    ok, out, err = run_freerouting(dpath, spath, jar=jar, max_passes=max_passes)
    if not ok:
        return {"ok": False, "routed_fraction": 0.0, "reward": fail_reward, "stderr": err[-400:]}
    with open(spath) as f:
        ses = f.read()
    m = parse_ses_routed_fraction(dsn, ses)
    m["ok"] = True
    m["reward"] = m["routed_fraction"]    # v0：先用 routed_fraction；之後可減 via/線長懲罰
    return m
