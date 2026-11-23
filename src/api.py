"""FastAPI entry point for the ticket-triage model."""

import asyncio
from contextlib import asynccontextmanager, suppress
import logging
import os
from pathlib import Path
from typing import Annotated

from anyio import to_thread
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .api_schemas import (
    ErrorResponse,
    LiveResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    ReadyResponse,
    ValidationErrorResponse,
    ValidationIssue,
)
from .bundle import ModelBundle, load_bundle
from .inference import describe_model, predict
from .request_limits import PredictionBodyLimit


logger = logging.getLogger(__name__)
MODEL_UNAVAILABLE = {503: {"model": ErrorResponse}}
PROBE_INTERVAL_SECONDS = 10
PROBE_TIMEOUT_SECONDS = 2
MAX_VALIDATION_ISSUES = 8
PROBE = PredictionRequest(
    description="Synthetic startup check",
    region="Europe",
    category="Infrastructure",
    priority="Medium",
    assigned_team="Global Operations",
    channel="Portal",
    backlog_at_creation=0,
    team_experience_years=3.0,
    previous_reassignments=0,
    sla_target_hours=36,
    created_at="2024-08-31T10:00:00Z",
)


async def monitor_model(app: FastAPI, bundle: ModelBundle, first_probe: asyncio.Event) -> None:
    while True:
        probe = asyncio.create_task(to_thread.run_sync(predict, bundle, PROBE, abandon_on_cancel=True))
        try:
            done, _ = await asyncio.wait({probe}, timeout=PROBE_TIMEOUT_SECONDS)
            if done:
                probe.result()
                app.state.model_ready = True
            else:
                app.state.model_ready = False
                logger.warning("Model readiness probe timed out.")
                first_probe.set()
                # Native prediction cannot be interrupted safely. Wait for this
                # attempt before scheduling another and discard its late result.
                await probe
        except Exception as exc:
            app.state.model_ready = False
            logger.warning("Model readiness probe failed (%s).", type(exc).__name__)
        finally:
            probe.cancel()
            first_probe.set()
        await asyncio.sleep(PROBE_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = None
    app.state.model_ready = False
    monitor = None
    bundle_path = os.environ.get("SLA_MODEL_BUNDLE")
    if bundle_path:
        try:
            # This path is configured by the operator. It is never taken from a request.
            bundle = load_bundle(Path(bundle_path).expanduser())
            info = describe_model(bundle)
        except Exception as exc:
            logger.error("Model bundle failed startup validation (%s).", type(exc).__name__)
        else:
            app.state.model_info = info
            app.state.bundle = bundle
            first_probe = asyncio.Event()
            monitor = asyncio.create_task(monitor_model(app, bundle, first_probe))
    else:
        logger.error("SLA_MODEL_BUNDLE is unset. Configure a trusted bundle and restart.")
    try:
        if monitor is not None:
            await first_probe.wait()
        yield
    finally:
        if monitor is not None:
            monitor.cancel()
            with suppress(asyncio.CancelledError):
                await monitor
        app.state.bundle = None
        app.state.model_ready = False


app = FastAPI(title="SLA Risk Inference", version="1.0.0", lifespan=lifespan)
app.add_middleware(PredictionBodyLimit)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    issues = []
    for error in exc.errors()[:MAX_VALIDATION_ISSUES]:
        location = error["loc"]
        field = location[1] if location[:1] == ("body",) and len(location) > 1 else None
        issues.append(ValidationIssue(
            field=field if field in PredictionRequest.model_fields else "body",
            code=error["type"],
            message="Invalid request field.",
        ))
    body = ValidationErrorResponse(detail=issues)
    return JSONResponse(status_code=422, content=body.model_dump())


def require_bundle(request: Request) -> ModelBundle:
    bundle = request.app.state.bundle
    if bundle is None or not request.app.state.model_ready:
        raise HTTPException(status_code=503, detail="Model unavailable.")
    return bundle


@app.get("/health/live")
async def check_liveness() -> LiveResponse:
    return LiveResponse()


@app.get("/health/ready", responses={503: {"model": ReadyResponse}})
async def check_readiness(request: Request, response: Response) -> ReadyResponse:
    if not request.app.state.model_ready:
        response.status_code = 503
        return ReadyResponse(status="not_ready")
    return ReadyResponse(status="ready")


@app.get("/v1/model/info", dependencies=[Depends(require_bundle)], responses=MODEL_UNAVAILABLE)
def model_info(request: Request) -> ModelInfoResponse:
    return request.app.state.model_info


@app.post(
    "/v1/predict",
    responses={
        **MODEL_UNAVAILABLE,
        413: {"model": ErrorResponse},
        422: {"model": ValidationErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def predict_ticket(
    ticket: PredictionRequest, bundle: Annotated[ModelBundle, Depends(require_bundle)]
) -> PredictionResponse:
    try:
        return predict(bundle, ticket)
    except Exception as exc:
        logger.error("Prediction failed (%s).", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Prediction failed.") from None
