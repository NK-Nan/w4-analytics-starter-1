"""test ของรายงานประจำเดือน (src/report.py)"""
import pytest

from src import metrics as m
from src import report


@pytest.fixture(scope="module")
def df():
    return m.load_orders()


def test_report_2026_03_net_sales(df, tmp_path):
    out = report.write_report("2026-03", tmp_path / "monthly_report.md", df=df)
    text = out.read_text(encoding="utf-8")
    assert "| ยอดขายสุทธิ | ฿1,147,155 |" in text


def test_report_has_sections(df):
    text = report.build_report(df, "2026-03")
    assert "เทียบเดือนก่อน" in text and "2026-02" in text
    assert "อัตรายกเลิก" in text and "อัตราคืนสินค้า" in text
    # หัวตาราง 2 แถว + สินค้า 5 แถว
    rows = [l for l in text.splitlines() if l.startswith("| ") and l.split("|")[1].strip().isdigit()]
    assert len(rows) == 5


def test_report_first_month_has_no_previous(df):
    text = report.build_report(df, "2025-10")
    assert "ไม่มีข้อมูลเดือน 2025-09" in text


def test_report_unknown_month(df):
    with pytest.raises(ValueError):
        report.build_report(df, "2030-01")


def test_cli_rejects_bad_month():
    assert report.main(["2026-13"]) == 2
