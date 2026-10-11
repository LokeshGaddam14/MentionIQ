from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.classifier import LLMError
from app.db import get_db
from app.models import Draft
from app.triage import triage_pending

router = APIRouter(tags=["review"])


class DraftOut(BaseModel):
    id: int
    mention_id: int
    status: str
    reply_text: str
    created_at: datetime
    reviewed_at: datetime | None
    mention_title: str
    mention_url: str
    urgency: int | None


class ApproveIn(BaseModel):
    reply_text: str | None = Field(default=None, min_length=10, max_length=700)


def to_out(d: Draft) -> DraftOut:
    return DraftOut(
        id=d.id,
        mention_id=d.mention_id,
        status=d.status,
        reply_text=d.reply_text,
        created_at=d.created_at,
        reviewed_at=d.reviewed_at,
        mention_title=d.mention.title,
        mention_url=d.mention.url,
        urgency=d.mention.urgency,
    )


@router.post("/triage")
def triage(min_urgency: int = 4, limit: int = 10, db: Session = Depends(get_db)) -> dict[str, int]:
    try:
        return triage_pending(db, min_urgency=max(1, min(min_urgency, 5)), limit=max(1, min(limit, 25)))
    except LLMError as e:
        raise HTTPException(502, f"LLM provider error: {e}")


@router.get("/drafts", response_model=list[DraftOut])
def list_drafts(status: str | None = "pending", db: Session = Depends(get_db)):
    q = select(Draft).options(joinedload(Draft.mention)).order_by(Draft.id.desc())
    if status:
        q = q.where(Draft.status == status)
    return [to_out(d) for d in db.scalars(q).all()]


def _review(db: Session, draft_id: int, status: str, text: str | None = None) -> DraftOut:
    d = db.get(Draft, draft_id)
    if not d:
        raise HTTPException(404, "Draft not found")
    if d.status != "pending":
        raise HTTPException(409, f"Draft already {d.status}")
    d.status = status
    d.reviewed_at = datetime.now(timezone.utc)
    if text:
        d.reply_text = text.strip()
    db.commit()
    return to_out(d)


@router.post("/drafts/{draft_id}/approve", response_model=DraftOut)
def approve(draft_id: int, body: ApproveIn | None = None, db: Session = Depends(get_db)):
    return _review(db, draft_id, "approved", body.reply_text if body else None)


@router.post("/drafts/{draft_id}/reject", response_model=DraftOut)
def reject(draft_id: int, db: Session = Depends(get_db)):
    return _review(db, draft_id, "rejected")
