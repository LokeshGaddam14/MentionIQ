from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base, get_db
from app.main import app
from app.models import Brand, Draft, Mention
from app.triage import DraftFailed, draft_reply, triage_pending

GOOD = '{"reply": "We are looking into this and will share an update soon."}'
no_sleep = lambda _s: None


def make_session():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def seed(db):
    now = datetime.now(timezone.utc)
    b = Brand(name="Acme", query="Acme")
    db.add(b)
    db.commit()
    db.add_all(
        [
            Mention(brand_id=b.id, source="t", url="u1", title="Acme breach", sentiment="negative", topic="security_privacy", urgency=5, classified_at=now),
            Mention(brand_id=b.id, source="t", url="u2", title="Acme wins award", sentiment="positive", topic="other", urgency=1, classified_at=now),
            Mention(brand_id=b.id, source="t", url="u3", title="Acme unclassified"),
        ]
    )
    db.commit()


def test_draft_reply_parses():
    assert "looking into" in draft_reply("Acme", "t", "s", llm=lambda s, u: GOOD, sleep=no_sleep)


def test_draft_reply_retries_on_bad_output():
    replies = iter(["nope", '{"reply": "x"}', GOOD])
    assert draft_reply("Acme", "t", "s", llm=lambda s, u: next(replies), sleep=no_sleep)


def test_draft_reply_gives_up():
    with pytest.raises(DraftFailed):
        draft_reply("Acme", "t", "s", llm=lambda s, u: "garbage", sleep=no_sleep)


def test_triage_only_urgent_and_idempotent():
    db = make_session()()
    seed(db)
    assert triage_pending(db, llm=lambda s, u: GOOD, sleep=no_sleep) == {
        "candidates": 1, "drafted": 1, "failed": 0,
    }
    assert triage_pending(db, llm=lambda s, u: GOOD, sleep=no_sleep)["drafted"] == 0
    assert db.query(Draft).count() == 1


def test_approve_then_conflict_and_reject():
    Session = make_session()
    db = Session()
    seed(db)
    triage_pending(db, llm=lambda s, u: GOOD, sleep=no_sleep)
    draft_id = db.query(Draft).one().id

    def override():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = override
    c = TestClient(app)
    try:
        r = c.post(f"/drafts/{draft_id}/approve", json={"reply_text": "Edited reply text here."})
        assert r.status_code == 200
        assert r.json()["status"] == "approved"
        assert r.json()["reply_text"] == "Edited reply text here."
        assert c.post(f"/drafts/{draft_id}/reject").status_code == 409
        assert c.get("/drafts").json() == []
        assert len(c.get("/drafts", params={"status": "approved"}).json()) == 1
        assert c.post("/drafts/999/approve").status_code == 404
    finally:
        app.dependency_overrides.clear()
