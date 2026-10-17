"""ฟังก์ชันคำนวณตัวชี้วัดของบ้านสบาย ช็อป

ทุกฟังก์ชันรับ DataFrame ที่ได้จาก load_orders() เป็นพารามิเตอร์แรก
นิยามตัวชี้วัดอยู่ใน CLAUDE.md
"""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

SUCCESS = "จัดส่งสำเร็จ"
CANCELLED = "ยกเลิก"
RETURNED = "คืนสินค้า"


def _success(df: pd.DataFrame) -> pd.DataFrame:
    """เฉพาะแถวที่จัดส่งสำเร็จ"""
    return df[df["status"] == SUCCESS]


def _between(values: pd.Series, start: str | None, end: str | None) -> pd.Series:
    """mask ของค่าที่อยู่ในช่วง start..end (รวมทั้งสองฝั่ง) ค่า None = ไม่จำกัดฝั่งนั้น"""
    mask = pd.Series(True, index=values.index)
    if start is not None:
        mask &= values >= start
    if end is not None:
        mask &= values <= end
    return mask


def _status_rate(df: pd.DataFrame, status: str) -> float:
    """จำนวนแถวสถานะนั้น ÷ จำนวนคำสั่งซื้อทั้งหมดใน df × 100"""
    if len(df) == 0:
        return 0.0
    return float((df["status"] == status).sum() / len(df) * 100)


def load_orders(path: Path | str = DATA / "orders.csv") -> pd.DataFrame:
    """อ่าน orders.csv แล้วเพิ่มคอลัมน์ที่ใช้บ่อย: net, unit_cost, gp, month, date

    net และ gp คำนวณทุกแถว (ไม่ดูสถานะ) ฟังก์ชันสรุปจะกรอง "จัดส่งสำเร็จ" เอง
    unit_cost มาจาก data/products.csv เสมอ แม้จะอ่านไฟล์ orders จากที่อื่น
    """
    orders = pd.read_csv(path, encoding="utf-8-sig", dtype={"order_id": str, "product_id": str})
    products = pd.read_csv(DATA / "products.csv", encoding="utf-8-sig", dtype={"product_id": str})
    df = orders.merge(products[["product_id", "unit_cost"]], on="product_id", how="left", validate="many_to_one")
    df["net"] = df["qty"] * df["unit_price"] * (1 - df["discount_pct"] / 100)
    df["gp"] = df["net"] - df["qty"] * df["unit_cost"]
    df["month"] = df["order_datetime"].astype(str).str[:7]
    df["date"] = df["order_datetime"].astype(str).str[:10]
    return df


def month_slice(df: pd.DataFrame, month: str) -> pd.DataFrame:
    """แถวทั้งหมด (ทุกสถานะ) ของเดือน 'YYYY-MM'"""
    return df[df["month"] == month]


def net_sales(df: pd.DataFrame) -> float:
    """ยอดขายสุทธิรวม (บาท)"""
    return float(_success(df)["net"].sum())


def gross_profit(df: pd.DataFrame) -> float:
    """กำไรขั้นต้นรวม (บาท)"""
    return float(_success(df)["gp"].sum())


def aov(df: pd.DataFrame) -> float:
    """ยอดขายสุทธิเฉลี่ยต่อคำสั่งซื้อที่จัดส่งสำเร็จ (บาท)"""
    orders = _success(df)["order_id"].nunique()
    return net_sales(df) / orders if orders else 0.0


def cancel_rate(df: pd.DataFrame) -> float:
    """อัตรายกเลิกของคำสั่งซื้อทั้งหมดใน df (เปอร์เซ็นต์ 0–100)"""
    return _status_rate(df, CANCELLED)


def monthly_sales(df: pd.DataFrame) -> pd.Series:
    """ยอดขายสุทธิรายเดือน index = 'YYYY-MM' เรียงตามเวลา"""
    return _success(df).groupby("month")["net"].sum().sort_index()


def best_month(df: pd.DataFrame) -> str:
    """เดือนที่ยอดขายสุทธิสูงสุด ('YYYY-MM')"""
    return str(monthly_sales(df).idxmax())


def channel_share(df: pd.DataFrame, channel: str, start_month: str, end_month: str) -> float:
    """สัดส่วนยอดขายสุทธิของช่องทางในช่วงเดือน start_month..end_month (รวมทั้งสองเดือน) เป็นเปอร์เซ็นต์"""
    period = _success(df)[_between(_success(df)["month"], start_month, end_month)]
    total = period["net"].sum()
    if total == 0:
        return 0.0
    return float(period.loc[period["channel"] == channel, "net"].sum() / total * 100)


def _ship_bucket(days: float) -> str:
    if days <= 2:
        return "1-2"
    if days <= 4:
        return "3-4"
    if days <= 6:
        return "5-6"
    return "7+"


def rating_by_ship_bucket(df: pd.DataFrame) -> dict[str, float]:
    """คะแนนรีวิวเฉลี่ยตามระยะส่ง คืน dict คีย์ '1-2', '3-4', '5-6', '7+'

    ใช้ทุกแถวที่มีทั้ง ship_days และ rating (ตัดค่าว่างก่อนเฉลี่ยตาม CLAUDE.md)
    """
    rated = df.dropna(subset=["ship_days", "rating"])
    means = rated.groupby(rated["ship_days"].map(_ship_bucket))["rating"].mean()
    return {k: float(means[k]) for k in ["1-2", "3-4", "5-6", "7+"] if k in means.index}


def top_province(df: pd.DataFrame, region: str) -> str:
    """จังหวัดที่มียอดขายสุทธิสูงสุดในภูมิภาคที่ระบุ"""
    s = _success(df)
    return str(s[s["region"] == region].groupby("province")["net"].sum().idxmax())


def top_products(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """n สินค้าที่ยอดขายสุทธิสูงสุดใน df คอลัมน์: product_id, product_name, qty, net"""
    return (
        _success(df)
        .groupby(["product_id", "product_name"], as_index=False)[["qty", "net"]]
        .sum()
        .sort_values(["net", "product_id"], ascending=[False, True])
        .head(n)
        .reset_index(drop=True)
    )


def avg_ship_days(df: pd.DataFrame, region: str | None = None, carrier: str | None = None,
                  start_date: str | None = None, end_date: str | None = None) -> float:
    """ระยะส่งเฉลี่ย (วัน) ตามตัวกรองที่ให้มา วันที่เป็น 'YYYY-MM-DD' และรวมทั้งสองวัน

    เทียบด้วยคอลัมน์ date (ไม่ใช่ order_datetime) เพื่อให้วันสุดท้ายถูกนับทั้งวัน
    """
    mask = _between(df["date"], start_date, end_date)
    if region is not None:
        mask &= df["region"] == region
    if carrier is not None:
        mask &= df["carrier"] == carrier
    return float(df.loc[mask, "ship_days"].dropna().mean())


def return_rate(df: pd.DataFrame, product_id: str | None = None,
                start_date: str | None = None, end_date: str | None = None) -> float:
    """อัตราคืนสินค้า (เปอร์เซ็นต์) ของสินค้าและช่วงวันที่ที่ระบุ

    ตัวหาร = คำสั่งซื้อทุกสถานะของสินค้าและช่วงนั้น
    """
    mask = _between(df["date"], start_date, end_date)
    if product_id is not None:
        mask &= df["product_id"] == product_id
    return _status_rate(df[mask], RETURNED)
