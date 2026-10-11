from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Brand(Base):
    __tablename__ = "brands"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    query: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    mentions: Mapped[list["Mention"]] = relationship(back_populates="brand")


class Mention(Base):
    __tablename__ = "mentions"
    __table_args__ = (UniqueConstraint("brand_id", "url", name="uq_mention_brand_url"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    brand_id: Mapped[int] = mapped_column(ForeignKey("brands.id"), index=True)
    source: Mapped[str] = mapped_column(String(40))
    url: Mapped[str] = mapped_column(String(1000))
    title: Mapped[str] = mapped_column(String(500))
    summary: Mapped[str] = mapped_column(Text, default="")
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # filled by the classifier in step 2
    sentiment: Mapped[str | None] = mapped_column(String(20), nullable=True)
    topic: Mapped[str | None] = mapped_column(String(60), nullable=True)
    urgency: Mapped[int | None] = mapped_column(Integer, nullable=True)
    classified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    brand: Mapped[Brand] = relationship(back_populates="mentions")
