"""单浮标潮位校时 API。

浮标重启会使毫秒计数归零，本服务把「开机段 + 段内计数」换算成可信的
UTC 毫秒时间线供潮位分析：

  POST /observations   分批提交观测（ID、开机段、计数、潮位）
  POST /calibrations   分批提交校时点（开机段、计数、UTC 毫秒）
  POST /docs           提交开机段 + 缓存清单，核对签收
  GET  /docs/{receipt_no}  按签收号查询已签收清单（也可 /docs?receipt_no=...）
  GET  /tide-levels    按 UTC 毫秒范围查可用潮位
  GET  /observations/{obs_id}  按观测 ID 查状态与原因
  GET  /docs           Swagger UI（不带 receipt_no 时）
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse
from pydantic import AliasChoices, BaseModel, Field

from . import core, db

app = FastAPI(
    title="单浮标潮位校时服务",
    description=__doc__,
    version="1.0.0",
    docs_url=None,  # /docs 同时承担 Swagger UI 与签收清单查询，见下方 docs()
)


# ---------------------------------------------------------------- 模型

class ObservationIn(BaseModel):
    obs_id: str = Field(min_length=1, description="观测 ID")
    boot_segment: str = Field(min_length=1, description="开机段")
    counter: int = Field(ge=0, description="段内毫秒计数")
    tide_level: float = Field(description="潮位")


class ObservationBatch(BaseModel):
    observations: list[ObservationIn] = Field(min_length=1)


class CalibrationIn(BaseModel):
    boot_segment: str = Field(min_length=1, description="开机段")
    counter: int = Field(ge=0, description="段内毫秒计数")
    utc_ms: int = Field(ge=0, description="对应的 UTC 毫秒")


class CalibrationBatch(BaseModel):
    calibrations: list[CalibrationIn] = Field(min_length=1)


class ChangedState(BaseModel):
    obs_id: str
    utc_ms: int | None
    status: str


class BatchResult(BaseModel):
    changed_count: int = Field(description="本次提交中时刻或状态变化的观测数")
    changed: list[ChangedState] = Field(description="时刻或状态变化的观测")


class TidePoint(BaseModel):
    obs_id: str
    utc_ms: int
    tide_level: float


class TideQueryResult(BaseModel):
    count: int
    items: list[TidePoint]


class CalibrationEndpoint(BaseModel):
    counter: int
    utc_ms: int


class CalibrationBasis(BaseModel):
    lo: CalibrationEndpoint = Field(description="下界校时点")
    hi: CalibrationEndpoint = Field(description="上界校时点")


class ManifestItem(BaseModel):
    obs_id: str = Field(
        min_length=1,
        description="观测 ID",
        validation_alias=AliasChoices("obs_id", "observation_id", "id"),
    )
    counter: int = Field(
        ge=0,
        description="段内计数",
        validation_alias=AliasChoices("counter", "count", "segment_count"),
    )
    tide_level: float = Field(
        description="潮位",
        validation_alias=AliasChoices("tide_level", "tide", "level"),
    )


ManifestItem.model_rebuild()


class ManifestItemIn(ManifestItem):
    boot_segment: str | None = Field(
        default=None,
        description="可选：逐项开机段；缺省沿用顶层开机段",
        validation_alias=AliasChoices("boot_segment", "segment"),
    )


class ReceiptRequest(BaseModel):
    boot_segment: str | None = Field(
        default=None,
        description="开机段；也可在清单每项中给出",
        validation_alias=AliasChoices("boot_segment", "segment", "boot"),
    )
    observations: list[ManifestItemIn] = Field(
        min_length=1,
        description="缓存清单；每项含观测 ID、段内计数和潮位",
        validation_alias=AliasChoices(
            "observations", "manifest", "items", "entries",
            "cache_manifest", "cache",
        ),
    )


class ReceiptResult(BaseModel):
    receipt_no: str = Field(description="签收号；同段内容相同的换序重提返回原号")
    boot_segment: str
    items: list[ManifestItem] = Field(description="已签收清单，可据此清除设备缓存")


class ObservationDetail(BaseModel):
    obs_id: str
    boot_segment: str
    counter: int
    tide_level: float
    utc_ms: int | None
    status: str = Field(description="ok / pending_calibration / conflict")
    reason: str | None
    calibration: CalibrationBasis | None = Field(description="校时依据；待校时为 null")


# ---------------------------------------------------------------- 提交

@app.post("/observations", response_model=BatchResult, summary="分批提交观测")
def submit_observations(batch: ObservationBatch) -> BatchResult:
    """写入观测原文并按当前校时点换算；同一 obs_id 重复提交会覆盖原文并重算。

    响应列出本次提交中时刻或状态发生变化的观测 ID（含新观测，
    以及因新观测落入同一 UTC 毫秒而被卷入/解除冲突的观测）。
    """
    conn = db.get_conn()
    with db.lock():
        with db.transaction(conn):
            try:
                ids = core.upsert_observations(
                    conn, [o.model_dump() for o in batch.observations]
                )
            except core.SealedRawError as exc:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "批内含与已签收原文不符的观测，整批拒绝",
                        "obs_id": exc.obs_id,
                        "reason": exc.detail,
                    },
                )
            changed = core.fetch_states(conn, ids)
    return BatchResult(changed_count=len(changed), changed=changed)


@app.post("/calibrations", response_model=BatchResult, summary="分批提交校时点")
def submit_calibrations(batch: CalibrationBatch) -> BatchResult:
    """写入校时点；同段同计数的新校时点替换旧值，并重算受影响观测。

    若合并后任一开机段内 UTC 随计数倒退，则整批拒绝（409），旧数据照常供数。
    响应列出本次提交中时刻或状态发生变化的观测 ID。
    """
    conn = db.get_conn()
    with db.lock():
        with db.transaction(conn):
            try:
                ids = core.upsert_calibrations(
                    conn, [c.model_dump() for c in batch.calibrations]
                )
            except core.MonotonicViolation as exc:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "校时点使段内 UTC 倒退，整批拒绝，旧数据照常供数",
                        "violations": exc.violations,
                    },
                )
            changed = core.fetch_states(conn, ids)
    return BatchResult(changed_count=len(changed), changed=changed)


# ---------------------------------------------------------------- 缓存清单签收

def _receipt_result(receipt: dict) -> ReceiptResult:
    return ReceiptResult(
        receipt_no=receipt["receipt_no"],
        boot_segment=receipt["boot_segment"],
        items=[
            ManifestItem(
                obs_id=it["obs_id"],
                counter=it["counter"],
                tide_level=it["tide_level"],
            )
            for it in receipt["items"]
        ],
    )


@app.post(
    "/docs",
    response_model=ReceiptResult,
    summary="提交开机段与缓存清单，核对签收",
)
def submit_receipt(req: ReceiptRequest) -> ReceiptResult:
    """核对一段的缓存清单，全部与岸站已收原文吻合才签收并返回签收号。

    - 每项须含观测 ID、段内计数和潮位；任一项缺收或不符则不签收，
      响应列出对应观测 ID，补传后可再次提交核对。
    - 一段只签收一份清单：内容相同（换序）重提返回原签收号，不新增签收；
      内容不同则拒绝（409）。
    - 签收后清单内观测受保护：`/observations` 只能原样重传，任一条原文
      不符整批拒绝（409）；补登校时点仍可改变 UTC 与供数状态。
    - 失败不新增签收或保护记录。
    """
    conn = db.get_conn()
    items = [it.model_dump() for it in req.observations]
    segment = req.boot_segment
    if segment is None:
        item_segments = {it["boot_segment"] for it in items}
        if None in item_segments or len(item_segments) > 1:
            raise HTTPException(
                status_code=422,
                detail="清单须属于同一开机段：请给出顶层 boot_segment",
            )
        segment = item_segments.pop()
    with db.lock():
        with db.transaction(conn):
            try:
                receipt = core.sign_receipt(conn, segment, items)
            except core.ReceiptMismatch as exc:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "清单与岸站已收原文不符，未签收",
                        "missing_obs_ids": exc.missing,
                        "mismatched_obs_ids": exc.mismatched,
                        "missing": exc.missing,
                        "mismatched": exc.mismatched,
                        "unmatched_obs_ids": exc.missing + exc.mismatched,
                        "issues": exc.issues,
                    },
                )
            except core.ReceiptConflict as exc:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "该开机段已签收内容不同的清单，拒绝新清单",
                        "receipt_no": exc.receipt_no,
                    },
                )
    return _receipt_result(receipt)


@app.get(
    "/docs/{receipt_no}",
    response_model=ReceiptResult,
    summary="按签收号查询已签收清单",
)
def get_receipt(receipt_no: str) -> ReceiptResult:
    """重启后仍可按签收号查询已签收清单（签收记录落库）。"""
    conn = db.get_conn()
    with db.lock():
        receipt = core.receipt_detail(conn, receipt_no)
    if receipt is None:
        raise HTTPException(status_code=404, detail=f"签收号 '{receipt_no}' 不存在")
    return _receipt_result(receipt)


@app.get("/docs", include_in_schema=False)
def docs(receipt_no: str | None = Query(default=None, description="签收号")):
    """不带签收号返回 Swagger UI；带 `receipt_no` 按其查询已签收清单。"""
    if receipt_no is None:
        return get_swagger_ui_html(openapi_url="/openapi.json", title="单浮标潮位校时服务")
    conn = db.get_conn()
    with db.lock():
        receipt = core.receipt_detail(conn, receipt_no)
    if receipt is None:
        raise HTTPException(status_code=404, detail=f"签收号 '{receipt_no}' 不存在")
    return JSONResponse(content=_receipt_result(receipt).model_dump())


@app.get(
    "/receipts/{receipt_no}",
    response_model=ReceiptResult,
    include_in_schema=False,
)
def get_receipt_alias(receipt_no: str) -> ReceiptResult:
    """`GET /docs/{receipt_no}` 的别名（便于不愿占用 /docs 路径的客户端）。"""
    return get_receipt(receipt_no)


# ---------------------------------------------------------------- 查询

@app.get("/tide-levels", response_model=TideQueryResult, summary="按 UTC 范围查可用潮位")
def tide_levels(
    start_ms: int = Query(ge=0, description="起始 UTC 毫秒（含）"),
    end_ms: int = Query(ge=0, description="结束 UTC 毫秒（含）"),
) -> TideQueryResult:
    """只返回已校时且无冲突（status=ok）的观测，按 UTC 毫秒升序。"""
    if end_ms < start_ms:
        raise HTTPException(status_code=422, detail="end_ms 不得小于 start_ms")
    conn = db.get_conn()
    with db.lock():
        rows = conn.execute(
            "SELECT obs_id, utc_ms, tide_level FROM observations "
            "WHERE status = ? AND utc_ms BETWEEN ? AND ? "
            "ORDER BY utc_ms, obs_id",
            (core.OK, start_ms, end_ms),
        ).fetchall()
    items = [
        TidePoint(obs_id=r["obs_id"], utc_ms=r["utc_ms"], tide_level=r["tide_level"])
        for r in rows
    ]
    return TideQueryResult(count=len(items), items=items)


@app.get(
    "/observations/{obs_id}",
    response_model=ObservationDetail,
    summary="按观测 ID 查状态和原因",
)
def get_observation(obs_id: str) -> ObservationDetail:
    conn = db.get_conn()
    with db.lock():
        row = conn.execute(
            "SELECT * FROM observations WHERE obs_id = ?", (obs_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"观测 '{obs_id}' 不存在")
    calibration = None
    if row["cal_lo_counter"] is not None:
        calibration = CalibrationBasis(
            lo=CalibrationEndpoint(
                counter=row["cal_lo_counter"], utc_ms=row["cal_lo_utc_ms"]
            ),
            hi=CalibrationEndpoint(
                counter=row["cal_hi_counter"], utc_ms=row["cal_hi_utc_ms"]
            ),
        )
    return ObservationDetail(
        obs_id=row["obs_id"],
        boot_segment=row["boot_segment"],
        counter=row["counter"],
        tide_level=row["tide_level"],
        utc_ms=row["utc_ms"],
        status=row["status"],
        reason=row["reason"],
        calibration=calibration,
    )


@app.get("/health", summary="健康检查")
def health() -> dict:
    return {"status": "up"}
