from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import ingest
from app.db import Base, get_db
from app.main import app


def make_client(monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    def override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    fake = [
        {"source": "google_news", "url": "https://x.test/1", "title": "A", "summary": "", "published_at": None},
        {"source": "google_news", "url": "https://x.test/1", "title": "A dup", "summary": "", "published_at": None},
        {"source": "google_news", "url": "https://x.test/2", "title": "B", "summary": "", "published_at": None},
    ]
    monkeypatch.setattr(ingest, "fetch_google_news", lambda q: fake)
    return TestClient(app)


def test_ingest_dedupes(monkeypatch):
    c = make_client(monkeypatch)
    b = c.post("/brands", json={"name": "Acme"}).json()
    assert c.post(f"/brands/{b['id']}/ingest").json() == {"new_mentions": 2}
    assert c.post(f"/brands/{b['id']}/ingest").json() == {"new_mentions": 0}
    assert len(c.get("/mentions", params={"brand_id": b["id"]}).json()) == 2


def test_duplicate_brand_conflict(monkeypatch):
    c = make_client(monkeypatch)
    assert c.post("/brands", json={"name": "Acme"}).status_code == 201
    assert c.post("/brands", json={"name": "Acme"}).status_code == 409
