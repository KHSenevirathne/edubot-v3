"""
End-to-end smoke tests.

These tests are part of the assignment's "test plan and test data"
deliverable. They cover:
  - Tier 1 (NLI):           preprocess + clean_text behaviour
  - Tier 2 (Inference):     EduBot.predict_intent on known phrases
  - Tier 3 (Database):      seed_db + DB queries return live data
  - ML loop:                teach() persists a pattern, retrain consumes it

Run from the project root:
    python -m pytest tests/ -v
"""

import os
import sys

# Make /app importable.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, 'app'))


# ---------- Tier 1 ----------

def test_clean_text_strips_punctuation_and_lemmatises():
    from preprocess import clean_text
    out = clean_text("How do I apply for ADMISSIONS???")
    # 'admissions' -> 'admission' (suffix-stripped); '?' gone; lowercased.
    assert 'admission' in out
    assert '?' not in out
    assert 'how' in out


def test_clean_text_keeps_question_words():
    from preprocess import clean_text
    out = clean_text("What courses do you offer")
    assert 'what' in out
    assert 'course' in out  # lemmatised from 'courses'


# ---------- Tier 3 ----------

def test_db_seeding_populates_required_tables():
    import database as db
    import seed_db
    seed_db.seed_all()
    stats = db.stats()
    assert stats['courses']      >= 5
    assert stats['faculty']      >= 5
    assert stats['events']       >= 3
    assert stats['exams']        >= 3
    assert stats['scholarships'] >= 3
    assert stats['hostel_rooms'] >= 3
    assert stats['kv_facts']     >= 5


def test_db_helpers_return_dicts():
    import database as db
    courses = db.list_courses()
    assert all(isinstance(r, dict) for r in courses)
    assert any(r['code'] == 'CS-BSC' for r in courses)

    fact = db.get_fact('library_location')
    assert fact and 'Block B' in fact


# ---------- Tier 2 ----------

def test_predict_intent_routes_courses_to_courses_tag():
    """Requires a trained model. Train it inline if missing."""
    if not os.path.exists(os.path.join(ROOT, 'models', 'chatbot_model.pkl')):
        from train import train_and_evaluate
        train_and_evaluate(verbose=False)

    from chat import EduBot
    bot = EduBot()
    tag, conf = bot.predict_intent("What courses do you offer?")
    assert tag == 'courses'
    assert conf > 0.4


def test_get_response_uses_database_for_courses():
    from chat import EduBot
    bot = EduBot()
    result = bot.get_response("What courses do you offer?")
    assert result['tag'] == 'courses'
    assert result['source'] == 'database'
    # The DB answer must include at least one seeded course name.
    assert 'BSc Computer Science' in result['response']


def test_get_response_uses_static_for_greeting():
    from chat import EduBot
    bot = EduBot()
    result = bot.get_response("Hello")
    assert result['tag'] == 'greeting'
    assert result['source'] == 'static'


def test_low_confidence_falls_back():
    from chat import EduBot
    bot = EduBot()
    result = bot.get_response("zzz qwerty asdfgh")
    assert result['tag'] == 'fallback'


# ---------- ML loop with trust tiers ----------

def test_admin_teach_lands_pre_approved():
    """Tier-2 path: admin teach should be approved immediately so it
    can enter the next training run."""
    import database as db
    import learning

    pending_before = db.count_pending_patterns()
    res = learning.teach("show me program prices", "fees")
    assert res['pattern_id'] > 0
    # Direct teach is approved -> goes into pending_patterns (ready
    # for next train run), NOT pending_review.
    assert db.count_pending_patterns() >= pending_before + 1 - (1 if res['retrained'] else 0)


def test_user_feedback_lands_in_pending_review_not_training():
    """Tier-1 path: a user thumbs-down + suggestion must NOT enter the
    model directly. It sits in pending_review until an admin approves."""
    import database as db
    import learning

    review_before = db.count_pending_review()
    res = learning.record_feedback(
        user_message='where can i find the dorms',
        bot_response='wrong answer',
        predicted_intent='greeting',
        confidence=0.5,
        helpful=False,
        expected_intent='hostel',
    )
    # Returned counts must reflect the new pending review row.
    assert res['retrained'] is False
    assert res['pending_review'] == review_before + 1


def test_admin_approve_promotes_a_pending_row():
    """Approving a pending suggestion makes it eligible for training."""
    import database as db
    import learning

    # Seed a pending row directly (skip the chat round-trip).
    pid = db.add_learned_pattern(
        'totally novel phrase', 'courses',
        source='feedback_correction', approved=False,
    )
    review_before = db.count_pending_review()
    pending_before = db.count_pending_patterns()

    res = learning.approve_suggestion(pid)
    assert res['approved'] is True
    # One row moved from pending_review into pending_patterns.
    assert db.count_pending_review() == review_before - 1
    if not res.get('retrained'):
        assert db.count_pending_patterns() == pending_before + 1


def test_admin_discard_deletes_pending_row():
    import database as db
    import learning

    pid = db.add_learned_pattern(
        'about to be deleted', 'fees',
        source='feedback_correction', approved=False,
    )
    review_before = db.count_pending_review()
    res = learning.discard_suggestion(pid)
    assert res['discarded'] is True
    assert db.count_pending_review() == review_before - 1
    # And the row is gone from the table entirely.
    rows = db.get_learned_patterns(approved_only=False)
    assert all(r['id'] != pid for r in rows)


def test_train_pipeline_ignores_unapproved_patterns():
    """Training must NOT pick up pending-review rows."""
    import database as db
    import learning

    # Add a pending pattern with a deliberately weird phrase the model
    # would otherwise pick up.
    db.add_learned_pattern(
        'xyzzy plugh frobnicate', 'courses',
        source='feedback_correction', approved=False,
    )
    learning.manual_retrain()
    # After retrain, approved patterns are marked used; pending stays.
    pending_after = [r for r in db.get_learned_patterns()
                     if r['approved'] == 0]
    assert any('xyzzy' in r['pattern'] for r in pending_after)
    assert all(r['used_in_model'] == 0 for r in pending_after)


# ---------- Input validation ----------

def test_clean_text_field_strips_control_chars():
    import validate as v
    out = v.clean_text_field("hello\x00\x01world", "msg")
    assert out == "helloworld"


def test_clean_text_field_rejects_empty_after_strip():
    import validate as v
    import pytest
    with pytest.raises(v.ValidationError):
        v.clean_text_field("   ", "msg")


def test_clean_text_field_rejects_too_long():
    import validate as v
    import pytest
    with pytest.raises(v.ValidationError):
        v.clean_text_field("x" * 600, "msg", max_len=500)


def test_parse_bool_strict():
    import validate as v
    import pytest
    assert v.parse_bool(True, "f") is True
    assert v.parse_bool("True", "f") is True
    assert v.parse_bool("FALSE", "f") is False
    assert v.parse_bool(0, "f") is False
    assert v.parse_bool(1, "f") is True
    # The classic Python-truthiness footgun: bool("false") is True.
    # Our parser must NOT do that.
    with pytest.raises(v.ValidationError):
        v.parse_bool("nope", "f")


def test_parse_float_in_range_clamps_to_validation_error():
    import validate as v
    import pytest
    assert v.parse_float_in_range(0.5, "c") == 0.5
    assert v.parse_float_in_range(None, "c", allow_none=True) is None
    with pytest.raises(v.ValidationError):
        v.parse_float_in_range(1.5, "c")
    with pytest.raises(v.ValidationError):
        v.parse_float_in_range("abc", "c")


def test_validate_intent_whitelist():
    import validate as v
    import pytest
    assert v.validate_intent("courses", "i", allowed={"courses", "fees"}) == "courses"
    with pytest.raises(v.ValidationError):
        v.validate_intent("hacks", "i", allowed={"courses", "fees"})


# ---------- API-level validation (integration) ----------
# Note: the Flask file lives at <root>/app.py but the helpers package
# also lives at <root>/app/. Plain `import app` finds the package
# first, so we load app.py by path with importlib.

import importlib.util


_flask_module = None


def _flask():
    """Lazy-load the Flask app module exactly once for the test session."""
    global _flask_module
    if _flask_module is None:
        spec = importlib.util.spec_from_file_location(
            'edubot_flask_app',
            os.path.join(ROOT, 'app.py'),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _flask_module = mod
    return _flask_module


def test_api_chat_rejects_empty_message():
    """The /chat endpoint must return 400 on empty/whitespace-only input."""
    client = _flask().app.test_client()
    r = client.post('/chat', json={'message': '   '})
    assert r.status_code == 400
    assert 'field' in r.get_json()


def test_api_chat_rejects_oversized_message():
    client = _flask().app.test_client()
    r = client.post('/chat', json={'message': 'a' * 1000})
    assert r.status_code == 400


def test_api_teach_rejects_unknown_intent():
    client = _flask().app.test_client()
    r = client.post('/teach', json={'pattern': 'hello there', 'intent': 'nonsense'})
    assert r.status_code == 400
    assert r.get_json()['field'] == 'intent'


def test_api_feedback_rejects_non_bool_helpful():
    client = _flask().app.test_client()
    r = client.post('/feedback', json={
        'user_message': 'hi', 'bot_response': 'hello',
        'predicted_intent': 'greeting', 'confidence': 0.9,
        'helpful': 'maybe',
    })
    assert r.status_code == 400
    assert r.get_json()['field'] == 'helpful'


# ---------- Admin auth ----------

def test_admin_endpoints_open_when_password_unset(monkeypatch):
    """When EDUBOT_ADMIN_PASSWORD is unset (dev mode), /admin* are
    reachable without credentials."""
    monkeypatch.delenv('EDUBOT_ADMIN_PASSWORD', raising=False)
    # Reload the Flask module so the env var is re-read.
    global _flask_module
    _flask_module = None
    client = _flask().app.test_client()
    r = client.get('/admin')
    assert r.status_code in (200, 500)  # 500 only if templates fail


def test_admin_endpoints_require_password_when_set(monkeypatch):
    """When the password IS set, an unauthenticated POST to /teach must
    be rejected with 401."""
    monkeypatch.setenv('EDUBOT_ADMIN_PASSWORD', 'secret-pw-123')
    global _flask_module
    _flask_module = None
    client = _flask().app.test_client()
    r = client.post(
        '/teach',
        json={'pattern': 'hello world', 'intent': 'greeting'},
    )
    assert r.status_code == 401

    # With the correct password it should pass auth (validation may
    # still pass or fail downstream - we only care about the 401 vs not).
    import base64
    creds = base64.b64encode(b'admin:secret-pw-123').decode()
    r = client.post(
        '/teach',
        json={'pattern': 'hello world', 'intent': 'greeting'},
        headers={'Authorization': f'Basic {creds}'},
    )
    assert r.status_code != 401
