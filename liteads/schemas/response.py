"""
API response schemas.
"""

from typing import Any

from pydantic import BaseModel, Field


class CreativeResponse(BaseModel):
    """Creative response schema."""

    title: str | None = Field(None, description="Ad title")
    description: str | None = Field(None, description="Ad description")
    image_url: str | None = Field(None, description="Image URL")
    video_url: str | None = Field(None, description="Video URL")
    landing_url: str = Field(..., description="Landing page URL")
    width: int | None = Field(None, description="Creative width")
    height: int | None = Field(None, description="Creative height")
    creative_type: str = Field(..., description="Creative type (banner/native/video)")


class TrackingUrls(BaseModel):
    """Tracking URLs for ad events."""

    impression_url: str = Field(..., description="URL to call on impression")
    click_url: str = Field(..., description="URL to call on click")
    conversion_url: str | None = Field(None, description="URL to call on conversion")


class AdResponse(BaseModel):
    """Single ad response schema."""

    ad_id: str = Field(..., description="Ad identifier")
    campaign_id: int = Field(..., description="Campaign identifier")
    creative_id: int = Field(..., description="Creative identifier")
    creative: CreativeResponse = Field(..., description="Creative content")
    tracking: TrackingUrls = Field(..., description="Tracking URLs")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")


class AdListResponse(BaseModel):
    """Ad list response schema."""

    request_id: str = Field(..., description="Request identifier")
    ads: list[AdResponse] = Field(default_factory=list, description="List of ads")
    count: int = Field(..., description="Number of ads returned")

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "req_abc123",
                "ads": [
                    {
                        "ad_id": "ad_456",
                        "campaign_id": 100,
                        "creative_id": 200,
                        "creative": {
                            "title": "Amazing Product",
                            "description": "Buy now!",
                            "image_url": "https://example.com/ad.jpg",
                            "landing_url": "https://example.com/landing",
                            "creative_type": "banner",
                        },
                        "tracking": {
                            "impression_url": "https://api.liteads.com/track/imp?id=xxx",
                            "click_url": "https://api.liteads.com/track/click?id=xxx",
                        },
                    }
                ],
                "count": 1,
            }
        }
    }


class EventResponse(BaseModel):
    """Event tracking response schema."""

    success: bool = Field(..., description="Whether the event was recorded")
    message: str | None = Field(None, description="Optional message")


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    database: bool = Field(..., description="Database connection status")
    redis: bool = Field(..., description="Redis connection status")


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: dict[str, Any] | None = Field(None, description="Error details")
    request_id: str | None = Field(None, description="Request identifier")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "validation_error",
                "message": "Invalid request parameters",
                "details": {"slot_id": "This field is required"},
                "request_id": "req_abc123",
            }
        }
    }
