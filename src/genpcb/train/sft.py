"""SFT warm-start：教生成器在 netlist 條件下吐合法 placement DSL。

任務形態 = placement：prompt（板框+元件宣告+netlist）→ completion（擺位），
completion-only loss。切分以 netlist 為單位防洩漏（data-engine.md §3）。

純函式（load_examples / split_by_netlist / config_hash）在模組層、本機可測；
torch/unsloth/trl 等重依賴鎖在 main() 內，僅 Colab/GPU 需要。

用法：
    python -m genpcb.train.sft --config configs/model_gemma4_12b.yaml
    python -m genpcb.train.sft --config configs/model_gemma4_12b.yaml --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

from genpcb.config import load_config
from genpcb.data.serialize import dsl_to_sft_example


def load_examples(path: str | Path) -> list[dict]:
    """讀 boards.jsonl → [{prompt, completion, netlist_sig}]。

    netlist_sig = prompt 的 hash；prompt 含元件宣告+netlist、無座標，故
    「同 netlist 的不同擺位變體」自然同 sig，切分時不會跨 train/val 洩漏。
    """
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = dsl_to_sft_example(json.loads(line)["text"])
            ex["netlist_sig"] = hashlib.sha1(ex["prompt"].encode("utf-8")).hexdigest()
            out.append(ex)
    return out


def split_by_netlist(examples: list[dict], val_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    sigs = sorted({e["netlist_sig"] for e in examples})
    random.Random(seed).shuffle(sigs)
    n_val = max(1, int(len(sigs) * val_frac)) if len(sigs) > 1 else 0
    val_sigs = set(sigs[:n_val])
    train = [e for e in examples if e["netlist_sig"] not in val_sigs]
    val = [e for e in examples if e["netlist_sig"] in val_sigs]
    return train, val


def config_hash(cfg: dict) -> str:
    """run 身分 = 模型 + 超參，排除儲存路徑（dataset/experiments），
    使 --dataset/--output-root 覆蓋不影響 run 名 → --resume 在 Colab 穩定。"""
    import copy
    c = copy.deepcopy(cfg)
    c.pop("paths", None)
    c.get("sft", {}).pop("dataset", None)
    return hashlib.sha1(json.dumps(c, sort_keys=True).encode("utf-8")).hexdigest()[:8]


def _latest_checkpoint(out_dir: Path) -> str | None:
    cks = list(out_dir.glob("checkpoint-*"))
    if not cks:
        return None
    return str(max(cks, key=lambda p: int(p.name.split("-")[1])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default=None, help="覆蓋 config 的資料集路徑（Colab：指向 Drive）")
    ap.add_argument("--output-root", default=None, help="覆蓋 experiments 根目錄（Colab：指向 Drive 以持久化 checkpoint）")
    ap.add_argument("--resume", action="store_true", help="從 output_dir 最新 checkpoint 續訓（Colab 斷線恢復）")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.dataset:
        cfg["sft"]["dataset"] = args.dataset
    if args.output_root:
        cfg["paths"]["experiments"] = args.output_root

    dataset_path = Path(cfg["sft"]["dataset"])
    if not dataset_path.exists():
        sys.exit(
            f"[sft] 資料集不存在：{dataset_path}\n"
            "先跑 `python -m genpcb.data.build` 或 notebooks/prepare_data.ipynb 產 D-sft。"
        )

    examples = load_examples(dataset_path)
    train, val = split_by_netlist(examples, cfg["sft"]["val_frac"], cfg["seed"])
    n_sigs = len({e["netlist_sig"] for e in examples})
    print(f"[sft] {len(examples)} examples, {n_sigs} netlists → train {len(train)} / val {len(val)}")

    # ── 以下為 GPU/訓練端；重依賴鎖在此 ──
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    from genpcb.models.adapter import ModelAdapter

    adapter = ModelAdapter(cfg)
    model, tokenizer = adapter.load_for_training()

    keep = lambda rows: Dataset.from_list([{"prompt": e["prompt"], "completion": e["completion"]} for e in rows])
    train_ds, val_ds = keep(train), (keep(val) if val else None)

    run = f"sft-{adapter.spec.family}-{config_hash(cfg)}"
    out_dir = Path(cfg["paths"]["experiments"]) / run
    sft = cfg["sft"]

    # NOTE: 對齊近期 TRL（completion_only_loss、max_length、eval_strategy）；
    # 若 Colab 上 TRL 版本參數名不同，煙霧測試會在此報錯，依訊息調整即可。
    sft_args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=sft["epochs"],
        learning_rate=float(sft["lr"]),
        per_device_train_batch_size=sft["per_device_batch"],
        gradient_accumulation_steps=sft["grad_accum"],
        max_length=adapter.spec.max_seq_len,
        completion_only_loss=True,                       # 只在擺位 completion 上算 loss
        logging_steps=sft["logging_steps"],
        eval_strategy=("steps" if val_ds else "no"),
        eval_steps=sft["eval_steps"],
        save_steps=sft["save_steps"],
        save_total_limit=sft["save_total_limit"],
        report_to=("wandb" if os.environ.get("WANDB_API_KEY") else "none"),
        run_name=run,
        seed=cfg["seed"],
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    resume = _latest_checkpoint(out_dir) if args.resume else None
    if resume:
        print(f"[sft] resume from {resume}")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(out_dir / "final"))
    print(f"[sft] done → {out_dir / 'final'}")


if __name__ == "__main__":
    main()
