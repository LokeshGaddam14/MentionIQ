from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import models  # noqa: F401  (registers tables)
from app.classifier import LLMError, classify_pending
from app.db import Base, engine, get_db
from app.ingest import ingest_all, ingest_brand
from app.models import Brand, Mention


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(engine)
    yield


app = FastAPI(title="MentionIQ", version="0.2.0", lifespan=lifespan)


class BrandIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    query: str | None = Field(default=None, max_length=255)


class BrandOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    query: str


class MentionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    brand_id: int
    source: str
    url: str
    title: str
    published_at: datetime | None
    sentiment: str | None
    topic: str | None
    urgency: int | None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/brands", response_model=BrandOut, status_code=201)
def create_brand(body: BrandIn, db: Session = Depends(get_db)) -> Brand:
    brand = Brand(name=body.name.strip(), query=(body.query or body.name).strip())
    db.add(brand)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Brand already exists")
    return brand


@app.get("/brands", response_model=list[BrandOut])
def list_brands(db: Session = Depends(get_db)):
    return db.scalars(select(Brand).order_by(Brand.name)).all()


@app.post("/brands/{brand_id}/ingest")
def ingest_one(brand_id: int, db: Session = Depends(get_db)) -> dict[str, int]:
    brand = db.get(Brand, brand_id)
    if not brand:
        raise HTTPException(404, "Brand not found")
    return {"new_mentions": ingest_brand(db, brand)}


@app.post("/ingest")
def ingest_everything(db: Session = Depends(get_db)) -> dict[str, int]:
    return ingest_all(db)


@app.post("/classify")
def classify(limit: int = 20, db: Session = Depends(get_db)) -> dict[str, int]:
    try:
        return classify_pending(db, limit=max(1, min(limit, 50)))
    except LLMError as e:
        raise HTTPException(502, f"LLM provider error: {e}")


@app.get("/mentions", response_model=list[MentionOut])
def list_mentions(
    brand_id: int | None = None,
    min_urgency: int | None = None,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    q = select(Mention).order_by(Mention.published_at.desc().nullslast()).limit(min(limit, 200))
    if brand_id:
        q = q.where(Mention.brand_id == brand_id)
    if min_urgency:
        q = q.where(Mention.urgency >= min_urgency)
    return db.scalars(q).all()
