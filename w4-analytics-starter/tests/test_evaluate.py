"""test ของ src/evaluate.py ใช้ผลทำนายสังเคราะห์จาก gold (ไม่เรียก API)"""
import pandas as pd
import pytest

from src import evaluate as ev


@pytest.fixture(scope="module")
def gold():
    return ev.load_gold()


def perfect_pred(gold, n=None):
    """ผลทำนายที่ตรงกับ gold ทุกฟิลด์ พร้อม usage สมมติ"""
    p = gold[["ticket_id", "category", "sentiment", "urgency", "order_id_mentioned", "contains_pii"]].copy()
    if n is not None:
        p = p.head(n)
    p["error"] = ""
    p["latency_s"] = "2.0"
    p["input_tokens"] = "1000"
    p["output_tokens"] = "100"
    p["cache_creation_input_tokens"] = "0"
    p["cache_read_input_tokens"] = "0"
    p["fallback_used"] = "False"
    return p.reset_index(drop=True)


def test_gold_has_100_tickets(gold):
    assert len(gold) == 100


def test_perfect_prediction(gold):
    r = ev.evaluate(perfect_pred(gold), gold, "claude-opus-5-5")
    assert r["matched"] == 100
    assert all(v == 1.0 for v in r["accuracy"].values())
    assert r["urgency_high_recall"] == 1.0
    assert r["mean_latency_s"] == pytest.approx(2.0)
    # ต่อใบ: 1000 × $4/M + 100 × $20/M = $0.006 → $6 ต่อ 1,000 ใบ
    assert r["cost_per_1000_usd"] == pytest.approx(6.0)
    cm = r["confusion"]
    assert int(cm.values.sum()) == 100
    assert int(cm.values.trace()) == 100


def test_wrong_and_error_rows(gold):
    p = perfect_pred(gold)
    high = p.index[p["urgency"] == "สูง"]
    normal = p.index[p["urgency"] == "ปกติ"]
    p.loc[high[0], "urgency"] = "ปกติ"            # พลาด urgency=สูง 1 ใบ
    p.loc[normal[0], "error"] = "rate_limited: x"  # error 1 ใบ (urgency ปกติ): ผิดทุกฟิลด์
    r = ev.evaluate(p, gold, "claude-opus-5-5")
    n_high = len(high)
    assert r["errors"] == 1
    assert r["accuracy"]["category"] == pytest.approx(0.99)
    assert r["accuracy"]["urgency"] == pytest.approx(0.98)
    assert r["urgency_high_recall"] == pytest.approx((n_high - 1) / n_high)
    assert ev.NO_RESULT in r["confusion"].columns


def test_partial_coverage_and_report(gold, tmp_path):
    p = perfect_pred(gold, n=5)
    r = ev.evaluate(p, gold, "claude-haiku-5-5")
    assert r["matched"] == 5
    md = ev.render_markdown([r])
    assert "| `claude-haiku-5-5` | 5/100 |" in md
    assert "ตรงกับ gold แค่ 5 จาก 100 ใบ" in md


def test_no_overlap(gold):
    p = perfect_pred(gold, n=3)
    p["ticket_id"] = ["X1", "X2", "X3"]
    r = ev.evaluate(p, gold, "claude-opus-5-5")
    assert r["matched"] == 0
    assert r["accuracy"]["category"] is None


def test_unknown_model_price(gold):
    r = ev.evaluate(perfect_pred(gold, n=2), gold, "some-other-model")
    assert r["cost_per_1000_usd"] is None
    assert "ไม่ทราบราคา" in ev.render_markdown([r])


def test_pii_bool_normalization(gold):
    p = perfect_pred(gold)
    p["contains_pii"] = p["contains_pii"].map({"True": "true", "False": "False"})
    r = ev.evaluate(p, gold, "claude-opus-5-5")
    assert r["accuracy"]["contains_pii"] == 1.0
