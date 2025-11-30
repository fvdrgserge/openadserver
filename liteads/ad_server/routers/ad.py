"""
Ad serving endpoints.
"""

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from liteads.ad_server.services.ad_service import AdService
from liteads.common.config import get_settings
from liteads.common.database import get_session
from liteads.common.logger import get_logger, log_context
from liteads.common.utils import generate_request_id
from liteads.schemas.request import AdRequest
from liteads.schemas.response import AdListResponse, AdResponse, CreativeResponse, TrackingUrls

logger = get_logger(__name__)
router = APIRouter()


def get_ad_service(session: AsyncSession = Depends(get_session)) -> AdService:
    """Dependency to get ad service."""
    return AdService(session)


@router.post("/request", response_model=AdListResponse)
async def request_ads(
    request: Request,
    ad_request: AdRequest,
    ad_service: AdService = Depends(get_ad_service),
) -> AdListResponse:
    """
    Request ads for a given slot.

    This is the main ad serving endpoint. It:
    1. Retrieves candidate ads based on targeting
    2. Filters ads by budget, frequency, and quality
    3. Predicts CTR/CVR using ML models
    4. Ranks ads by eCPM
    5. Returns top ads with tracking URLs
    """
    request_id = generate_request_id()
    settings = get_settings()

    # Add request context for logging
    log_context(
        request_id=request_id,
        slot_id=ad_request.slot_id,
        user_id=ad_request.user_id,
    )

    logger.info(
        "Ad request received",
        num_requested=ad_request.num_ads,
        os=ad_request.device.os if ad_request.device else None,
    )

    # Get client IP from request
    client_ip = request.client.host if request.client else None
    if ad_request.geo and not ad_request.geo.ip:
        ad_request.geo.ip = client_ip

    # Serve ads
    candidates = await ad_service.serve_ads(
        request=ad_request,
        request_id=request_id,
    )

    # Build response
    ads = []
    base_url = str(request.base_url).rstrip("/")

    for candidate in candidates[: ad_request.num_ads]:
        # Build tracking URLs
        tracking = TrackingUrls(
            impression_url=f"{base_url}/api/v1/event/track?type=impression&req={request_id}&ad={candidate.campaign_id}",
            click_url=f"{base_url}/api/v1/event/track?type=click&req={request_id}&ad={candidate.campaign_id}",
            conversion_url=f"{base_url}/api/v1/event/track?type=conversion&req={request_id}&ad={candidate.campaign_id}",
        )

        # Build creative response
        creative = CreativeResponse(
            title=candidate.title,
            description=candidate.description,
            image_url=candidate.image_url,
            video_url=candidate.video_url,
            landing_url=candidate.landing_url,
            width=candidate.width,
            height=candidate.height,
            creative_type=_get_creative_type_name(candidate.creative_type),
        )

        ad = AdResponse(
            ad_id=f"ad_{candidate.campaign_id}_{candidate.creative_id}",
            campaign_id=candidate.campaign_id,
            creative_id=candidate.creative_id,
            creative=creative,
            tracking=tracking,
            metadata={
                "ecpm": round(candidate.ecpm, 4),
                "pctr": round(candidate.pctr, 6),
            }
            if settings.debug
            else None,
        )
        ads.append(ad)

    logger.info(
        "Ad request completed",
        num_returned=len(ads),
    )

    return AdListResponse(
        request_id=request_id,
        ads=ads,
        count=len(ads),
    )


def _get_creative_type_name(creative_type: int) -> str:
    """Convert creative type enum to string."""
    types = {1: "banner", 2: "native", 3: "video", 4: "interstitial"}
    return types.get(creative_type, "banner")
