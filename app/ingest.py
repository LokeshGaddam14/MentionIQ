from datetime import datetime, timezone
from urllib.parse import quote_plus

import feedparser
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Brand, Mention

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"


def _parse_dt(entry) -> datetime | None:
    t = getattr(entry, "published_parsed", None)
    return datetime(*t[:6], tzinfo=timezone.utc) if t else None


def fetch_google_news(query: str) -> list[dict]:
    feed = feedparser.parse(GOOGLE_NEWS_RSS.format(q=quote_plus(query)))
    return [
        {
            "source": "google_news",
            "url": e.link,
            "title": e.title[:500],
            "summary": getattr(e, "summary", ""),
            "published_at": _parse_dt(e),
        }
        for e in feed.entries
        if getattr(e, "link", None) and getattr(e, "title", None)
    ]


def ingest_brand(db: Session, brand: Brand) -> int:
    items = fetch_google_news(brand.query)
    if not items:
        return 0
    existing = set(
        db.scalars(select(Mention.url).where(Mention.brand_id == brand.id)).all()
    )
    new = [i for i in items if i["url"] not in existing]
    # dedupe within the batch itself
    seen: set[str] = set()
    unique = []
    for i in new:
        if i["url"] not in seen:
            seen.add(i["url"])
            unique.append(i)
    db.add_all(Mention(brand_id=brand.id, **i) for i in unique)
    db.commit()
    return len(unique)


def ingest_all(db: Session) -> dict[str, int]:
    return {b.name: ingest_brand(db, b) for b in db.scalars(select(Brand)).all()}
