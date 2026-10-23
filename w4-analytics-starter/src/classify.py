"""จัดหมวด ticket ลูกค้าด้วย Claude API ทีละใบ

ใช้งาน:
    export ANTHROPIC_API_KEY=...            # อ่าน key จาก environment variable เท่านั้น
    python -m src.classify --model claude-opus-5-5 --limit 5

- system prompt = prompts/classify_v3.md
- บังคับรูปแบบผลด้วย structured outputs (output_config.format แบบ json_schema)
- เขียนผลลง out/pred_<model>.csv พร้อม usage และเวลาต่อใบ
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROMPT_PATH = ROOT / "prompts" / "classify_v3.md"
OUT_DIR = ROOT / "out"

CATEGORIES = [
    "ส่งช้า",
    "ขอคืนเงิน",
    "สินค้าชำรุด",
    "ได้รับสินค้าผิด",
    "เปลี่ยนแปลงคำสั่งซื้อ",
    "ชำระเงินและโปรโมชั่น",
    "สอบถามทั่วไป",
    "ชมเชย",
]
SENTIMENTS = ["บวก", "กลาง", "ลบ"]
URGENCIES = ["ปกติ", "สูง"]

SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": CATEGORIES},
        "sentiment": {"type": "string", "enum": SENTIMENTS},
        "urgency": {"type": "string", "enum": URGENCIES},
        "order_id_mentioned": {
            "type": "string",
            "description": "รหัสคำสั่งซื้อที่ลูกค้าพิมพ์มาในข้อความ ตามที่เขียนไว้ ถ้าไม่มีให้เป็นสตริงว่าง",
        },
        "contains_pii": {
            "type": "boolean",
            "description": "true ถ้าข้อความมีข้อมูลส่วนบุคคล เช่น เบอร์โทร อีเมล ที่อยู่ เลขบัตร",
        },
    },
    "required": ["category", "sentiment", "urgency", "order_id_mentioned", "contains_pii"],
    "additionalProperties": False,
}
FIELDS = list(SCHEMA["properties"])

# โมเดลที่รับ fallbacks: "default" ได้ (ถ้าถูก safety classifier ปฏิเสธ จะรันซ้ำบนโมเดลสำรองฝั่งเซิร์ฟเวอร์)
FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# ข้อความในไฟล์ prompt ตั้งต้นของ repo (ยังไม่ได้วาง prompt v3)
PLACEHOLDER_MARK = "วาง prompt v3"

OUT_COLUMNS = (
    ["ticket_id", *FIELDS]
    + ["model", "stop_reason", "fallback_used", "input_tokens", "output_tokens",
       "cache_creation_input_tokens", "cache_read_input_tokens", "latency_s", "error"]
)


def load_prompt(path: Path | None = None) -> str:
    """อ่าน system prompt และหยุดถ้ายังเป็นไฟล์ตั้งต้นที่ไม่มี prompt จริง"""
    path = path or PROMPT_PATH
    text = path.read_text(encoding="utf-8").strip()
    if not text or (PLACEHOLDER_MARK in text and len(text) < 200):
        raise SystemExit(
            f"{path} ยังไม่มี prompt v3 (มีแค่ข้อความตั้งต้น) "
            "กรุณาวาง prompt v3 จากคาบ 2 ลงในไฟล์นี้ก่อนรัน"
        )
    return text


def make_client() -> anthropic.Anthropic:
    """สร้าง client จาก ANTHROPIC_API_KEY ใน environment เท่านั้น"""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ไม่พบ ANTHROPIC_API_KEY ใน environment กรุณา export ANTHROPIC_API_KEY=... ก่อนรัน")
    return anthropic.Anthropic(api_key=key)


def safe_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def classify_one(client: anthropic.Anthropic, model: str, system_prompt: str, message: str,
                 use_fallback: bool, effort: str | None) -> dict:
    """ส่ง ticket 1 ใบ คืน dict ที่มีฟิลด์ผลลัพธ์ + usage + เวลา"""
    params = dict(
        model=model,
        max_tokens=4096,  # เผื่อ thinking ของโมเดลที่ปิด thinking ไม่ได้ ผลจริงสั้นมาก
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": message}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    )
    if effort:
        params["output_config"]["effort"] = effort

    start = time.perf_counter()
    if use_fallback:
        response = client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **params)
    else:
        response = client.messages.create(**params)
    latency = time.perf_counter() - start

    usage = response.usage
    row = {
        "model": response.model,
        "stop_reason": response.stop_reason,
        "fallback_used": any(getattr(i, "type", None) == "fallback_message" for i in (getattr(usage, "iterations", None) or [])),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
        "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
        "latency_s": round(latency, 3),
        "error": "",
    }
    if response.stop_reason == "refusal":
        row["error"] = "refusal"
        return row
    if response.stop_reason == "max_tokens":
        row["error"] = "max_tokens"
        return row
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        row["error"] = "no_text_block"
        return row
    row.update(json.loads(text))
    return row


def run(model: str, input_path: Path, limit: int | None, use_fallback: bool,
        effort: str | None, client: anthropic.Anthropic | None = None) -> Path:
    system_prompt = load_prompt()
    client = client or make_client()
    with open(input_path, encoding="utf-8-sig", newline="") as f:
        tickets = list(csv.DictReader(f))
    if limit is not None:
        tickets = tickets[:limit]

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"pred_{safe_name(model)}.csv"
    use_fallback = use_fallback and model in FALLBACK_MODELS

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for i, t in enumerate(tickets, start=1):
            row = {"ticket_id": t["ticket_id"]}
            try:
                row.update(classify_one(client, model, system_prompt, t["message"], use_fallback, effort))
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError) as e:
                # key ผิด / ไม่มีสิทธิ์ / ชื่อโมเดลผิด: ทุกใบจะพังเหมือนกัน หยุดเลย
                raise SystemExit(f"หยุดที่ {t['ticket_id']}: {type(e).__name__}: {e.message}")
            except anthropic.RateLimitError as e:
                row["error"] = f"rate_limited: {e.message}"
            except anthropic.APIStatusError as e:
                row["error"] = f"api_error {e.status_code}: {e.message}"
            except anthropic.APIConnectionError as e:
                row["error"] = f"connection_error: {e}"
            except json.JSONDecodeError as e:
                row["error"] = f"invalid_json: {e}"
            writer.writerow(row)
            f.flush()
            status = row["error"] or f"{row.get('category')} · {row.get('sentiment')} · {row.get('urgency')}"
            print(f"[{i}/{len(tickets)}] {t['ticket_id']} {status} ({row.get('latency_s', '-')} s)")
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="จัดหมวด ticket ด้วย Claude API")
    p.add_argument("--model", required=True, help="ชื่อโมเดล เช่น claude-opus-5-5")
    p.add_argument("--input", type=Path, default=DATA / "tickets.csv")
    p.add_argument("--limit", type=int, default=5, help="จำนวนใบแรกที่จะรัน (ค่าเริ่มต้น 5, ใส่ 0 = ทั้งหมด)")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default=None,
                   help="output_config.effort (ไม่ใส่ = ค่าเริ่มต้นของโมเดล)")
    p.add_argument("--no-fallback", action="store_true",
                   help="ไม่ใช้ fallbacks: \"default\" (ผลทุกใบมาจากโมเดลที่ระบุเท่านั้น)")
    args = p.parse_args(argv)
    out = run(args.model, args.input, args.limit or None, not args.no_fallback, args.effort)
    print(f"เขียนผลแล้ว: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
