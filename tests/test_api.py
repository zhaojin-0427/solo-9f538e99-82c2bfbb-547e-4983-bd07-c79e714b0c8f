"""端到端测试：校时换算、替换重算、冲突与恢复、整批拒绝、变更清单、查询。"""

import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_tables():
    conn = db.get_conn()
    with db.lock():
        conn.execute("DELETE FROM receipt_items")
        conn.execute("DELETE FROM receipts")
        conn.execute("DELETE FROM observations")
        conn.execute("DELETE FROM calibration_points")
    yield


# ---------------------------------------------------------------- 工具

def post_obs(items):
    r = client.post("/observations", json={"observations": items})
    assert r.status_code == 200, r.text
    return r.json()


def post_cal(items, expect=200):
    r = client.post("/calibrations", json={"calibrations": items})
    assert r.status_code == expect, r.text
    return r.json()


def get_obs(obs_id):
    return client.get(f"/observations/{obs_id}")


def tide(start, end):
    r = client.get("/tide-levels", params={"start_ms": start, "end_ms": end})
    assert r.status_code == 200, r.text
    return r.json()


def changed_ids(resp):
    return {c["obs_id"] for c in resp["changed"]}


# ---------------------------------------------------------------- 校时换算

def test_pending_without_calibration_keeps_raw():
    resp = post_obs([
        {"obs_id": "o1", "boot_segment": "A", "counter": 100, "tide_level": 1.5},
    ])
    assert changed_ids(resp) == {"o1"}
    body = get_obs("o1").json()
    assert body["status"] == "pending_calibration"
    assert body["utc_ms"] is None
    assert body["calibration"] is None
    # 原文保留
    assert (body["boot_segment"], body["counter"], body["tide_level"]) == ("A", 100, 1.5)
    assert "尚无校时点" in body["reason"]


def test_linear_interpolation_with_floor_and_basis():
    post_cal([
        {"boot_segment": "A", "counter": 1000, "utc_ms": 10000},
        {"boot_segment": "A", "counter": 2000, "utc_ms": 20050},
    ])
    resp = post_obs([
        {"obs_id": "lo", "boot_segment": "A", "counter": 1000, "tide_level": 1.0},
        {"obs_id": "floor", "boot_segment": "A", "counter": 1001, "tide_level": 2.0},
        {"obs_id": "mid", "boot_segment": "A", "counter": 1500, "tide_level": 3.0},
        {"obs_id": "hi", "boot_segment": "A", "counter": 2000, "tide_level": 4.0},
        {"obs_id": "above", "boot_segment": "A", "counter": 2001, "tide_level": 5.0},
        {"obs_id": "below", "boot_segment": "A", "counter": 999, "tide_level": 6.0},
    ])
    assert changed_ids(resp) == {"lo", "floor", "mid", "hi", "above", "below"}

    assert get_obs("lo").json()["utc_ms"] == 10000
    # 10000 + 1*10050/1000 = 10010.05 -> 向下取整 10010
    assert get_obs("floor").json()["utc_ms"] == 10010
    # 10000 + 500*10050/1000 = 15025
    mid = get_obs("mid").json()
    assert mid["utc_ms"] == 15025 and mid["status"] == "ok"
    assert mid["calibration"] == {
        "lo": {"counter": 1000, "utc_ms": 10000},
        "hi": {"counter": 2000, "utc_ms": 20050},
    }
    assert get_obs("hi").json()["utc_ms"] == 20050

    for oid in ("above", "below"):
        body = get_obs(oid).json()
        assert body["status"] == "pending_calibration"
        assert body["utc_ms"] is None
        assert "超出" in body["reason"]


def test_single_calibration_point_anchors_exact_counter_only():
    post_cal([{"boot_segment": "A", "counter": 500, "utc_ms": 7777}])
    post_obs([
        {"obs_id": "hit", "boot_segment": "A", "counter": 500, "tide_level": 1.0},
        {"obs_id": "miss", "boot_segment": "A", "counter": 501, "tide_level": 2.0},
    ])
    assert get_obs("hit").json()["utc_ms"] == 7777
    assert get_obs("hit").json()["status"] == "ok"
    assert get_obs("miss").json()["status"] == "pending_calibration"


# ---------------------------------------------------------------- 替换与重算

def test_replaced_calibration_recalculates_affected():
    post_cal([
        {"boot_segment": "A", "counter": 1000, "utc_ms": 10000},
        {"boot_segment": "A", "counter": 2000, "utc_ms": 20000},
    ])
    post_obs([{"obs_id": "x", "boot_segment": "A", "counter": 1500, "tide_level": 1.0}])
    assert get_obs("x").json()["utc_ms"] == 15000

    # 同段同计数的新校时点替换旧值 -> 重算
    resp = post_cal([{"boot_segment": "A", "counter": 2000, "utc_ms": 21000}])
    assert changed_ids(resp) == {"x"}
    assert resp["changed"][0]["utc_ms"] == 15500
    assert get_obs("x").json()["utc_ms"] == 15500


def test_late_calibration_extends_range_and_fills_pending():
    post_cal([
        {"boot_segment": "A", "counter": 1000, "utc_ms": 10000},
        {"boot_segment": "A", "counter": 2000, "utc_ms": 20000},
    ])
    post_obs([{"obs_id": "y", "boot_segment": "A", "counter": 3000, "tide_level": 1.0}])
    assert get_obs("y").json()["status"] == "pending_calibration"

    # 补传校时点 -> 区间外观测被重算为可用
    resp = post_cal([{"boot_segment": "A", "counter": 3000, "utc_ms": 30000}])
    assert changed_ids(resp) == {"y"}
    body = get_obs("y").json()
    assert body["status"] == "ok" and body["utc_ms"] == 30000


def test_calibration_batch_without_observations_changes_nothing():
    resp = post_cal([{"boot_segment": "A", "counter": 0, "utc_ms": 100}])
    assert resp["changed_count"] == 0 and resp["changed"] == []


# ---------------------------------------------------------------- 整批拒绝

def test_backwards_utc_rejects_whole_batch_and_keeps_old_data():
    post_cal([
        {"boot_segment": "A", "counter": 1000, "utc_ms": 10000},
        {"boot_segment": "A", "counter": 2000, "utc_ms": 20000},
    ])
    post_obs([{"obs_id": "z", "boot_segment": "A", "counter": 1500, "tide_level": 2.5}])
    assert get_obs("z").json()["utc_ms"] == 15000

    # 同一批里：B 段校时点本身合法，A 段新点使 UTC 倒退 -> 整批拒绝
    resp = post_cal([
        {"boot_segment": "B", "counter": 0, "utc_ms": 5000},
        {"boot_segment": "A", "counter": 1500, "utc_ms": 9000},
    ], expect=409)
    assert "整批拒绝" in resp["detail"]["message"]
    assert resp["detail"]["violations"] == [{
        "boot_segment": "A", "counter": 1500, "utc_ms": 9000,
        "prev_counter": 1000, "prev_utc_ms": 10000,
    }]

    # 旧数据照常供数：A 段观测不变，B 段合法点也未落库
    assert get_obs("z").json()["utc_ms"] == 15000
    assert tide(0, 99999)["count"] == 1
    post_obs([{"obs_id": "b1", "boot_segment": "B", "counter": 0, "tide_level": 1.0}])
    assert get_obs("b1").json()["status"] == "pending_calibration"


def test_equal_utc_within_segment_is_not_backwards():
    post_cal([
        {"boot_segment": "A", "counter": 0, "utc_ms": 1000},
        {"boot_segment": "A", "counter": 10, "utc_ms": 1000},  # 持平不算倒退
    ])
    post_obs([{"obs_id": "e", "boot_segment": "A", "counter": 5, "tide_level": 1.0}])
    assert get_obs("e").json()["utc_ms"] == 1000


# ---------------------------------------------------------------- 冲突与恢复

def test_conflict_across_boot_segments_and_restore():
    post_cal([
        {"boot_segment": "A", "counter": 0, "utc_ms": 1000},
        {"boot_segment": "B", "counter": 0, "utc_ms": 1000},
    ])
    post_obs([{"obs_id": "a1", "boot_segment": "A", "counter": 0, "tide_level": 1.0}])
    assert get_obs("a1").json()["status"] == "ok"

    # 不同开机段落在同一 UTC 毫秒 -> 双方均不供分析
    resp = post_obs([{"obs_id": "b1", "boot_segment": "B", "counter": 0, "tide_level": 2.0}])
    assert changed_ids(resp) == {"a1", "b1"}
    for oid in ("a1", "b1"):
        body = get_obs(oid).json()
        assert body["status"] == "conflict"
        assert "冲突" in body["reason"]
    assert tide(0, 99999)["count"] == 0

    # 更正校时点解除冲突 -> 双方恢复
    resp = post_cal([{"boot_segment": "B", "counter": 0, "utc_ms": 2000}])
    assert changed_ids(resp) == {"a1", "b1"}
    assert get_obs("a1").json()["status"] == "ok"
    b1 = get_obs("b1").json()
    assert b1["status"] == "ok" and b1["utc_ms"] == 2000
    assert tide(0, 99999)["count"] == 2


def test_same_segment_same_ms_is_not_conflict():
    post_cal([{"boot_segment": "A", "counter": 0, "utc_ms": 1000}])
    post_obs([
        {"obs_id": "s1", "boot_segment": "A", "counter": 0, "tide_level": 1.0},
        {"obs_id": "s2", "boot_segment": "A", "counter": 0, "tide_level": 2.0},
    ])
    assert get_obs("s1").json()["status"] == "ok"
    assert get_obs("s2").json()["status"] == "ok"


def test_calibration_change_can_conflict_other_segment():
    post_cal([
        {"boot_segment": "A", "counter": 0, "utc_ms": 1000},
        {"boot_segment": "B", "counter": 0, "utc_ms": 5000},
    ])
    post_obs([
        {"obs_id": "a1", "boot_segment": "A", "counter": 0, "tide_level": 1.0},
        {"obs_id": "b1", "boot_segment": "B", "counter": 0, "tide_level": 2.0},
    ])
    assert get_obs("a1").json()["status"] == "ok"

    # 更正 B 段校时点，使 b1 落到 a1 的毫秒上 -> a1 被卷入冲突并列进变更清单
    resp = post_cal([{"boot_segment": "B", "counter": 0, "utc_ms": 1000}])
    assert changed_ids(resp) == {"a1", "b1"}
    assert get_obs("a1").json()["status"] == "conflict"
    assert get_obs("b1").json()["status"] == "conflict"
    assert tide(0, 99999)["count"] == 0


def test_overwritten_observation_can_resolve_conflict():
    post_cal([
        {"boot_segment": "A", "counter": 0, "utc_ms": 1000},
        {"boot_segment": "B", "counter": 0, "utc_ms": 1000},
        {"boot_segment": "B", "counter": 10, "utc_ms": 5000},
    ])
    post_obs([
        {"obs_id": "a1", "boot_segment": "A", "counter": 0, "tide_level": 1.0},
        {"obs_id": "b1", "boot_segment": "B", "counter": 0, "tide_level": 2.0},
    ])
    assert get_obs("a1").json()["status"] == "conflict"

    # 覆盖 b1 原文使其移走 -> a1 自动恢复
    resp = post_obs([{"obs_id": "b1", "boot_segment": "B", "counter": 10, "tide_level": 2.0}])
    assert changed_ids(resp) == {"a1", "b1"}
    assert get_obs("a1").json()["status"] == "ok"
    assert get_obs("b1").json()["utc_ms"] == 5000


# ---------------------------------------------------------------- 变更清单与查询

def test_identical_resubmission_reports_no_change():
    item = {"obs_id": "r1", "boot_segment": "A", "counter": 0, "tide_level": 1.0}
    post_cal([{"boot_segment": "A", "counter": 0, "utc_ms": 1000}])
    assert changed_ids(post_obs([item])) == {"r1"}
    assert changed_ids(post_obs([item])) == set()  # 无变化不列出


def test_tide_levels_range_query_only_serves_ok():
    post_cal([
        {"boot_segment": "A", "counter": 0, "utc_ms": 1000},
        {"boot_segment": "A", "counter": 100, "utc_ms": 1100},
        {"boot_segment": "B", "counter": 0, "utc_ms": 1050},
        {"boot_segment": "C", "counter": 0, "utc_ms": 1050},
    ])
    post_obs([
        {"obs_id": "in1", "boot_segment": "A", "counter": 0, "tide_level": 1.0},
        {"obs_id": "in2", "boot_segment": "A", "counter": 40, "tide_level": 2.0},
        {"obs_id": "edge", "boot_segment": "A", "counter": 100, "tide_level": 3.0},
        {"obs_id": "cf1", "boot_segment": "B", "counter": 0, "tide_level": 4.0},
        {"obs_id": "cf2", "boot_segment": "C", "counter": 0, "tide_level": 5.0},
        {"obs_id": "pend", "boot_segment": "A", "counter": 500, "tide_level": 6.0},
    ])
    body = tide(1000, 1100)  # 边界含端点；冲突与待校时不供分析
    assert [p["obs_id"] for p in body["items"]] == ["in1", "in2", "edge"]
    assert body["items"][1]["utc_ms"] == 1040
    assert tide(1051, 1051)["count"] == 0
    assert tide(0, 999)["count"] == 0


def test_tide_levels_rejects_inverted_range():
    r = client.get("/tide-levels", params={"start_ms": 2000, "end_ms": 1000})
    assert r.status_code == 422


def test_get_observation_404_and_health():
    assert get_obs("nope").status_code == 404
    assert client.get("/health").json() == {"status": "up"}


def test_docs_and_openapi_available():
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/observations", "/calibrations", "/tide-levels", "/observations/{obs_id}"):
        assert p in paths


# ---------------------------------------------------------------- 缓存清单签收

MANIFEST = [
    {"obs_id": "c1", "counter": 100, "tide_level": 1.25},
    {"obs_id": "c2", "counter": 200, "tide_level": 2.5},
    {"obs_id": "c3", "counter": 300, "tide_level": 3.75},
]


def seed_segment(items=None, segment="A"):
    post_obs([
        {"obs_id": it["obs_id"], "boot_segment": segment,
         "counter": it["counter"], "tide_level": it["tide_level"]}
        for it in (items or MANIFEST)
    ])


def post_receipt(manifest, segment="A", expect=200):
    r = client.post("/docs", json={"boot_segment": segment, "observations": manifest})
    assert r.status_code == expect, r.text
    return r.json()


def test_receipt_signs_after_full_match():
    seed_segment()
    body = post_receipt(MANIFEST)
    assert body["receipt_no"].startswith("RC-")
    assert body["boot_segment"] == "A"
    assert [it["obs_id"] for it in body["items"]] == ["c1", "c2", "c3"]
    assert {it["counter"] for it in body["items"]} == {100, 200, 300}
    assert {it["tide_level"] for it in body["items"]} == {1.25, 2.5, 3.75}


def test_receipt_lists_missing_and_mismatched_ids():
    seed_segment(MANIFEST[:2])
    manifest = MANIFEST[:2] + [{"obs_id": "c9", "counter": 900, "tide_level": 9.0}]
    body = post_receipt(manifest, expect=409)
    assert body["detail"]["missing"] == ["c9"]
    assert body["detail"]["missing_obs_ids"] == ["c9"]
    assert body["detail"]["mismatched"] == []

    seed_segment([MANIFEST[0], {**MANIFEST[1], "tide_level": 99.0}])
    body = post_receipt(MANIFEST[:2], expect=409)
    assert body["detail"]["mismatched"] == ["c2"]
    assert body["detail"]["missing"] == []
    # 未签收，也没有保护记录：不符观测仍可被补传覆盖
    post_obs([{"obs_id": "c2", "boot_segment": "A", "counter": 200,
               "tide_level": 2.5}])
    assert post_receipt(MANIFEST[:2])["receipt_no"].startswith("RC-")


def test_receipt_retry_after_supplement_upload():
    seed_segment(MANIFEST[:2])
    # 清单只列已收的两条即全部吻合，可以签收
    first = post_receipt(MANIFEST[:2])
    # 补传第三条后，同段换内容清单被拒绝，原签收号不变
    seed_segment(MANIFEST)
    body = post_receipt(MANIFEST, expect=409)
    assert body["detail"]["receipt_no"] == first["receipt_no"]


def test_receipt_reorder_returns_same_no_and_changed_content_rejected():
    seed_segment()
    first = post_receipt(MANIFEST)
    reordered = list(reversed(MANIFEST))
    again = post_receipt(reordered)
    assert again["receipt_no"] == first["receipt_no"]
    assert [it["obs_id"] for it in again["items"]] == ["c1", "c2", "c3"]

    changed = [{**it, "tide_level": it["tide_level"] + 0.5} for it in MANIFEST]
    body = post_receipt(changed, expect=409)
    assert body["detail"]["receipt_no"] == first["receipt_no"]
    # 只留一份清单
    assert client.get(f"/docs/{first['receipt_no']}").json()["items"] == first["items"]


def test_receipt_protects_raw_but_allows_calibration_updates():
    post_cal([
        {"boot_segment": "A", "counter": 0, "utc_ms": 1000},
        {"boot_segment": "A", "counter": 1000, "utc_ms": 11000},
    ])
    seed_segment()
    receipt = post_receipt(MANIFEST)
    assert get_obs("c2").json()["utc_ms"] == 3000

    # 原样重传整批可接受
    resp = post_obs([
        {"obs_id": it["obs_id"], "boot_segment": "A",
         "counter": it["counter"], "tide_level": it["tide_level"]}
        for it in MANIFEST
    ])
    assert changed_ids(resp) == set()

    # 批内任一条与已签收原文不符 -> 整批拒绝
    bad = [
        {"obs_id": "c1", "boot_segment": "A", "counter": 100, "tide_level": 1.25},
        {"obs_id": "c2", "boot_segment": "A", "counter": 200, "tide_level": 2.5001},
        {"obs_id": "c3", "boot_segment": "A", "counter": 300, "tide_level": 3.75},
    ]
    r = client.post("/observations", json={"observations": bad})
    assert r.status_code == 409
    assert r.json()["detail"]["obs_id"] == "c2"
    # 原文未被改写
    c2 = get_obs("c2").json()
    assert (c2["counter"], c2["tide_level"]) == (200, 2.5)

    # 补登校时点仍可改变 UTC 与供数状态
    resp = post_cal([{"boot_segment": "A", "counter": 1000, "utc_ms": 21000}])
    assert changed_ids(resp) == {"c1", "c2", "c3"}
    assert get_obs("c2").json()["utc_ms"] == 5000
    assert tide(0, 99999)["count"] == 3

    # 签收记保持不变
    assert client.get(f"/docs/{receipt['receipt_no']}").json()["items"] == receipt["items"]


def test_receipt_rejects_observation_from_other_segment():
    seed_segment(MANIFEST[:2])
    seed_segment(MANIFEST[2:], segment="B")
    body = post_receipt(MANIFEST, expect=409)
    assert body["detail"]["mismatched"] == ["c3"]
    assert client.get("/docs/RC-NOPE").status_code == 404


def test_receipt_persists_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("BUOY_DB", str(tmp_path / "buoy.db"))
    db.reset_for_tests()
    try:
        seed_segment()
        receipt = post_receipt(MANIFEST)
        db.reset_for_tests()  # 模拟重启：丢弃连接后重新打开同一库
        for url in (
            f"/docs/{receipt['receipt_no']}",
            None,
        ):
            if url is None:
                r = client.get("/docs", params={"receipt_no": receipt["receipt_no"]})
            else:
                r = client.get(url)
            assert r.status_code == 200
            assert r.json()["items"] == receipt["items"]
        # 重启后保护仍生效
        r = client.post("/observations", json={"observations": [
            {"obs_id": "c1", "boot_segment": "A", "counter": 100,
             "tide_level": 1.25001},
        ]})
        assert r.status_code == 409
    finally:
        db.reset_for_tests()
        monkeypatch.setenv("BUOY_DB", ":memory:")
