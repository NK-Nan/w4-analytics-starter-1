"""วัดผลการจัดหมวด ticket เทียบกับ data/tickets_test_gold.csv (100 ใบ)

ใช้งาน:
    python -m src.evaluate                      # ทุกไฟล์ out/pred_*.csv
    python -m src.evaluate out/pred_claude-opus-5-5.csv

รายงานต่อโมเดล: accuracy ทุกฟิลด์, recall ของ urgency=สูง, confusion matrix ของ category,
เวลาเฉลี่ยต่อใบ และต้นทุนต่อ 1,000 ใบ แล้วเขียนตารางเทียบลง reports/model_comparison.md
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from src.classify import CATEGORIES, FIELDS, OUT_DIR

ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = ROOT / "data" / "tickets_test_gold.csv"
REPORT_PATH = ROOT / "reports" / "model_comparison.md"
NO_RESULT = "(ไม่มีผล)"

# ราคาต่อ 1 ล้าน token (USD) ของ Claude API ณ ต.ค. 2026 แก้ได้ถ้าราคาเปลี่ยน
# cache_write = เขียน cache แบบ 5 นาที (1.25 เท่าของ input) · cache_read = อ่านจาก cache
PRICES = {
    "claude-fable-5-1": {"input": 10.00, "output": 50.00, "cache_read": 0.25},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}


def load_gold(path: Path = GOLD_PATH) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def load_pred(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def model_from_path(path: Path) -> str:
    return path.stem.removeprefix("pred_")


def _norm(series: pd.Series, field: str) -> pd.Series:
    """ทำค่าให้เทียบกันได้: ตัดช่องว่าง และแปลง contains_pii เป็น true/false"""
    s = series.fillna("").astype(str).str.strip()
    if field == "contains_pii":
        s = s.str.lower().map({"true": "true", "1": "true", "false": "false", "0": "false"}).fillna("")
    return s


def _join(pred: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """gold ที่มีผลทำนาย (ticket_id ตรงกัน) คอลัมน์ <field>_gold และ <field>_pred"""
    cols = ["ticket_id", *FIELDS, "error"]
    p = pred.reindex(columns=cols, fill_value="").drop_duplicates("ticket_id", keep="last")
    m = gold[["ticket_id", *FIELDS]].merge(p, on="ticket_id", how="inner", suffixes=("_gold", "_pred"))
    for f in FIELDS:
        m[f + "_gold"] = _norm(m[f + "_gold"], f)
        m[f + "_pred"] = _norm(m[f + "_pred"], f)
        # แถวที่เรียก API ไม่สำเร็จ ถือว่าทำนายผิดทุกฟิลด์
        m.loc[m["error"] != "", f + "_pred"] = NO_RESULT
    return m


def confusion(pred: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """confusion matrix ของ category แถว = gold, คอลัมน์ = ผลทำนาย"""
    m = _join(pred, gold)
    cols = CATEGORIES + ([NO_RESULT] if (m["category_pred"] == NO_RESULT).any() else [])
    other = sorted(set(m["category_pred"]) - set(cols))  # ค่าที่ไม่อยู่ใน enum (ไม่ควรเกิด)
    table = pd.crosstab(m["category_gold"], m["category_pred"])
    return table.reindex(index=CATEGORIES, columns=cols + other, fill_value=0)


def cost_usd(pred: pd.DataFrame, model: str) -> float | None:
    """ต้นทุนรวมของทุกแถวในไฟล์ผล (USD) คืน None ถ้าไม่รู้ราคาของโมเดล"""
    price = PRICES.get(model)
    if price is None:
        return None
    tok = {c: pd.to_numeric(pred.get(c, 0), errors="coerce").fillna(0).sum()
           for c in ["input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"]}
    cache_read = price.get("cache_read", price["input"] * 0.1)
    return (
        tok["input_tokens"] * price["input"]
        + tok["cache_creation_input_tokens"] * price["input"] * 1.25
        + tok["cache_read_input_tokens"] * cache_read
        + tok["output_tokens"] * price["output"]
    ) / 1_000_000


def evaluate(pred: pd.DataFrame, gold: pd.DataFrame, model: str) -> dict:
    """สรุปผลของโมเดลเดียว"""
    m = _join(pred, gold)
    n = len(m)
    acc = {f: float((m[f + "_gold"] == m[f + "_pred"]).mean()) if n else None for f in FIELDS}
    high = m[m["urgency_gold"] == "สูง"]
    ok = pred["error"] == "" if "error" in pred else pd.Series(True, index=pred.index)
    latency = pd.to_numeric(pred.loc[ok, "latency_s"], errors="coerce").dropna() if "latency_s" in pred else pd.Series(dtype=float)
    total_cost = cost_usd(pred, model)
    rows = len(pred)
    fallback = (pred.get("fallback_used", pd.Series(dtype=str)).astype(str).str.lower() == "true").sum()
    return {
        "model": model,
        "matched": n,
        "gold_total": len(gold),
        "pred_rows": rows,
        "errors": int((m["error"] != "").sum()),
        "accuracy": acc,
        "urgency_high_recall": float((high["urgency_pred"] == "สูง").mean()) if len(high) else None,
        "urgency_high_n": len(high),
        "mean_latency_s": float(latency.mean()) if len(latency) else None,
        "cost_per_1000_usd": total_cost / rows * 1000 if total_cost is not None and rows else None,
        "fallback_rows": int(fallback),
        "confusion": confusion(pred, gold),
    }


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def render_markdown(results: list[dict]) -> str:
    lines = [
        "# เทียบผลการจัดหมวด ticket ระหว่างโมเดล",
        "",
        f"วัดกับ `data/tickets_test_gold.csv` ({results[0]['gold_total'] if results else 100} ใบ) "
        "นับเฉพาะ ticket ที่อยู่ทั้งในไฟล์ผลและ gold แถวที่เรียก API ไม่สำเร็จนับว่าผิดทุกฟิลด์",
        "",
        "| โมเดล | ใบที่วัด | error | category | sentiment | urgency | recall urgency=สูง | order_id | contains_pii "
        "| เวลาเฉลี่ย (วินาที) | ต้นทุน / 1,000 ใบ (USD) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        a = r["accuracy"]
        lat = "—" if r["mean_latency_s"] is None else f"{r['mean_latency_s']:.2f}"
        cost = "ไม่ทราบราคา" if r["cost_per_1000_usd"] is None else f"${r['cost_per_1000_usd']:.2f}"
        lines.append(
            f"| `{r['model']}` | {r['matched']}/{r['gold_total']} | {r['errors']} | {_pct(a['category'])} "
            f"| {_pct(a['sentiment'])} | {_pct(a['urgency'])} | {_pct(r['urgency_high_recall'])} "
            f"| {_pct(a['order_id_mentioned'])} | {_pct(a['contains_pii'])} | {lat} | {cost} |"
        )

    notes = [r for r in results if r["matched"] < r["gold_total"]]
    lines += ["", "## หมายเหตุ", ""]
    for r in notes:
        lines.append(f"- `{r['model']}` มีผลตรงกับ gold แค่ {r['matched']} จาก {r['gold_total']} ใบ "
                     "ตัวเลขยังเทียบกับโมเดลอื่นไม่ได้ ให้รัน classify กับ test ให้ครบก่อน")
    for r in results:
        if r["fallback_rows"]:
            lines.append(f"- `{r['model']}` มี {r['fallback_rows']} ใบที่ตอบโดยโมเดลสำรอง (fallback) "
                         "ต้นทุนของใบเหล่านั้นคิดด้วยราคาโมเดลหลัก จึงเป็นค่าประมาณ")
    lines += [
        "- ต้นทุนต่อ 1,000 ใบ = ต้นทุนรวมของทุกแถวในไฟล์ผล ÷ จำนวนแถว × 1,000 ใช้ราคาใน `PRICES` ของ `src/evaluate.py`",
        "- เวลาเฉลี่ยนับเฉพาะแถวที่เรียก API สำเร็จ",
        "- `order_id_mentioned` ใน gold เป็นรหัสคำสั่งซื้อที่ผูกกับ ticket ไม่ใช่รหัสที่ลูกค้าพิมพ์ "
        "ฟิลด์นี้จึงทำได้ไม่เต็มจากข้อความอย่างเดียว",
        "",
    ]

    for r in results:
        cm = r["confusion"]
        lines += [f"## Confusion matrix: `{r['model']}`", "", "แถว = gold · คอลัมน์ = ผลทำนาย", ""]
        lines.append("| gold \\ ทำนาย | " + " | ".join(cm.columns) + " |")
        lines.append("|---|" + "---:|" * len(cm.columns))
        for cat, row in cm.iterrows():
            lines.append(f"| {cat} | " + " | ".join(str(int(v)) for v in row) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="วัดผลการจัดหมวด ticket เทียบกับ gold")
    p.add_argument("pred_files", nargs="*", type=Path, help="ไฟล์ผล (ค่าเริ่มต้น: out/pred_*.csv ทั้งหมด)")
    p.add_argument("--out", type=Path, default=REPORT_PATH)
    args = p.parse_args(argv)

    files = args.pred_files or sorted(OUT_DIR.glob("pred_*.csv"))
    if not files:
        print(f"ไม่พบไฟล์ผลใน {OUT_DIR} ให้รัน python -m src.classify ก่อน", file=sys.stderr)
        return 1

    gold = load_gold()
    results = [evaluate(load_pred(f), gold, model_from_path(f)) for f in files]
    for r in results:
        a = r["accuracy"]
        print(f"{r['model']}: วัด {r['matched']}/{r['gold_total']} ใบ · category {_pct(a['category'])} "
              f"· recall สูง {_pct(r['urgency_high_recall'])}")
        if r["matched"] == 0:
            print("  ไม่มี ticket ในไฟล์นี้ตรงกับ gold (5 ใบแรกของ tickets.csv ไม่อยู่ใน test) "
                  "ให้รัน: python -m src.classify --model <ชื่อ> --input ../data/tickets_test.csv --limit 0")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(results), encoding="utf-8")
    print(f"เขียนตารางเทียบแล้ว: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
