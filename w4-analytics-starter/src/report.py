"""สร้างรายงานประจำเดือน reports/monthly_report.md

ใช้งาน: python -m src.report 2026-03

ตัวเลขทุกตัวคำนวณผ่านฟังก์ชันใน src/metrics.py เท่านั้น
"""
import re
import sys
from pathlib import Path

import pandas as pd

from src import metrics as m

ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "reports" / "monthly_report.md"


def previous_month(month: str) -> str:
    """เดือนก่อนหน้าของ 'YYYY-MM'"""
    return str(pd.Period(month, freq="M") - 1)


def build_report(df: pd.DataFrame, month: str) -> str:
    """คืนเนื้อหารายงาน Markdown ของเดือน 'YYYY-MM'"""
    current = m.month_slice(df, month)
    if current.empty:
        raise ValueError(f"ไม่มีข้อมูลของเดือน {month}")

    prev = previous_month(month)
    prev_df = m.month_slice(df, prev)
    net = m.net_sales(current)
    if prev_df.empty:
        compare = f"ไม่มีข้อมูลเดือน {prev} ให้เทียบ"
    else:
        prev_net = m.net_sales(prev_df)
        if prev_net:
            change = (net / prev_net - 1) * 100
            compare = f"{change:+.2f}% จากเดือน {prev} (฿{prev_net:,.0f})"
        else:
            compare = f"เดือน {prev} ไม่มียอดขายสุทธิ"

    lines = [
        f"# รายงานยอดขายประจำเดือน {month}",
        "",
        "| ตัวชี้วัด | ค่า |",
        "|---|---|",
        f"| ยอดขายสุทธิ | ฿{net:,.0f} |",
        f"| เทียบเดือนก่อน | {compare} |",
        f"| อัตรายกเลิก | {m.cancel_rate(current):.2f}% |",
        f"| อัตราคืนสินค้า | {m.return_rate(current):.2f}% |",
        "",
        "## 5 สินค้าขายดี (ตามยอดขายสุทธิ)",
        "",
        "| อันดับ | รหัส | สินค้า | จำนวนชิ้น | ยอดขายสุทธิ |",
        "|---|---|---|---:|---:|",
    ]
    for rank, row in enumerate(m.top_products(current, 5).itertuples(index=False), start=1):
        lines.append(f"| {rank} | {row.product_id} | {row.product_name} | {int(row.qty):,} | ฿{row.net:,.0f} |")
    lines += [
        "",
        "นิยามตาม CLAUDE.md: ยอดขายสุทธิ = qty × unit_price × (1 − discount_pct/100) เฉพาะ \"จัดส่งสำเร็จ\" "
        "อัตรายกเลิก/คืน = จำนวนแถวสถานะนั้น ÷ คำสั่งซื้อทั้งหมดของเดือน × 100",
        "",
    ]
    return "\n".join(lines)


def write_report(month: str, path: Path = REPORT_PATH, df: pd.DataFrame | None = None) -> Path:
    """เขียนรายงานลงไฟล์ แล้วคืน path ของไฟล์"""
    if df is None:
        df = m.load_orders()
    text = build_report(df, month)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", args[0]):
        print("วิธีใช้: python -m src.report YYYY-MM  เช่น python -m src.report 2026-03", file=sys.stderr)
        return 2
    try:
        path = write_report(args[0])
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"สร้างรายงานแล้ว: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
