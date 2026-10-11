import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.classifier import (
    ClassificationFailed,
    LLMTransient,
    build_user_prompt,
    classify_pending,
    classify_text,
)
from app.db import Base
from app.models import Brand, Mention

GOOD = '{"sentiment": "negative", "topic": "security_privacy", "urgency": 5}'
no_sleep = lambda _s: None


def make_db():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)()


def test_valid_output_parsed():
    c = classify_text("Breach at Acme", "", llm=lambda s, u: GOOD, sleep=no_sleep)
    assert (c.sentiment, c.topic, c.urgency) == ("negative", "security_privacy", 5)


def test_retries_on_invalid_json_then_succeeds():
    replies = iter(["not json at all", '{"sentiment": "angry"}', GOOD])
    c = classify_text("x", "y", llm=lambda s, u: next(replies), sleep=no_sleep)
    assert c.urgency == 5


def test_retries_on_transient_error():
    calls = {"n": 0}

    def flaky(s, u):
        calls["n"] += 1
        if calls["n"] < 3:
            raise LLMTransient("429")
        return GOOD

    assert classify_text("x", "y", llm=flaky, sleep=no_sleep).topic == "security_privacy"


def test_gives_up_after_retries():
    with pytest.raises(ClassificationFailed):
        classify_text("x", "y", llm=lambda s, u: "garbage", sleep=no_sleep)


def test_out_of_range_urgency_rejected():
    bad = '{"sentiment": "neutral", "topic": "other", "urgency": 9}'
    with pytest.raises(ClassificationFailed):
        classify_text("x", "y", llm=lambda s, u: bad, sleep=no_sleep)


def test_prompt_strips_html_and_wraps_text():
    p = build_user_prompt("Title", "<p>hello   <b>world</b></p>")
    assert "<b>" not in p and "hello world" in p and p.startswith("<mention>")


def test_classify_pending_updates_db_and_counts_failures():
    db = make_db()
    b = Brand(name="Acme", query="Acme")
    db.add(b)
    db.commit()
    db.add_all(
        [
            Mention(brand_id=b.id, source="t", url="u1", title="good"),
            Mention(brand_id=b.id, source="t", url="u2", title="bad"),
        ]
    )
    db.commit()

    def llm(system, user):
        return GOOD if "good" in user else "garbage"

    result = classify_pending(db, llm=llm, sleep=no_sleep)
    assert result == {"classified": 1, "failed": 1, "remaining": 0}
    m = db.query(Mention).filter_by(url="u1").one()
    assert m.urgency == 5 and m.classified_at is not None
    assert db.query(Mention).filter_by(url="u2").one().classified_at is None
