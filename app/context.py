"""
context.py - Session-based dialogue state for EduBot v3.

Holds a small per-session memory so the bot can carry a real
"consulting" conversation across turns: resolving pronouns ("it",
"this one", "that course") to the previously discussed entity, and
remembering which programme the user is currently asking about.

This is the dialogue-management layer that the brief calls for under
"Effective implementation of NLP" - intent classification on its own
treats every message as independent.

State is in-memory and per-process (fine for a single-instance Flask
deployment; for multi-worker production you would back this with
Redis or the database).
"""

import re
import time
import uuid
from threading import Lock


# Anaphora trigger phrases. Whole-word, case-insensitive.
# Order matters - longer phrases first so "this course" matches before "this".
_PRONOUN_PATTERNS = [
    re.compile(r'\bthis course\b',  re.IGNORECASE),
    re.compile(r'\bthat course\b',  re.IGNORECASE),
    re.compile(r'\bthe course\b',   re.IGNORECASE),
    re.compile(r'\bthis program(?:me)?\b', re.IGNORECASE),
    re.compile(r'\bthat program(?:me)?\b', re.IGNORECASE),
    re.compile(r'\bthe program(?:me)?\b',  re.IGNORECASE),
    re.compile(r'\bthis one\b',     re.IGNORECASE),
    re.compile(r'\bthat one\b',     re.IGNORECASE),
    re.compile(r'\bit\b',           re.IGNORECASE),
]


# Short list of phrases that imply the user wants to act on a
# previously-mentioned item, even without a pronoun (e.g. "price of it"
# but also "the fees" right after a course detail). Used as a hint when
# deciding whether to apply the remembered entity.
_FOLLOWUP_HINTS = re.compile(
    r'\b(price|fees|cost|tuition|admission|apply|enroll|details|more|tell me more)\b',
    re.IGNORECASE,
)


SESSION_TTL_SECONDS = 60 * 60         # idle sessions are wiped after 1h
MAX_HISTORY_TURNS    = 10
MAX_SESSIONS         = 1000           # rough cap to avoid unbounded growth


_SESSIONS = {}
_LOCK = Lock()


def new_session_id():
    """Generate a new opaque session ID for the frontend to remember."""
    return uuid.uuid4().hex


def get_session(session_id, create=True):
    """Return the per-session state dict, creating it if needed.

    Returns None when session_id is falsy and create=False.
    """
    if not session_id:
        if not create:
            return None
        session_id = new_session_id()

    now = time.time()
    with _LOCK:
        sess = _SESSIONS.get(session_id)
        if sess is None or now - sess['last_seen'] > SESSION_TTL_SECONDS:
            sess = {
                'id':           session_id,
                'last_entity':  None,    # canonical course name, or None
                'last_intent':  None,
                'history':      [],
                'last_seen':    now,
                'pending_clarification': None,  # 'course' when we asked which one
            }
            _SESSIONS[session_id] = sess
        sess['last_seen'] = now

        # Cheap LRU-ish eviction: if we've blown past the cap, drop the
        # oldest sessions. Keeps memory bounded on a long-running server.
        if len(_SESSIONS) > MAX_SESSIONS:
            oldest = sorted(
                _SESSIONS.items(), key=lambda kv: kv[1]['last_seen']
            )[: len(_SESSIONS) - MAX_SESSIONS]
            for k, _ in oldest:
                _SESSIONS.pop(k, None)

        return sess


def has_pronoun(text):
    """True if the text contains an anaphoric reference like 'it' / 'this one'."""
    return any(p.search(text) for p in _PRONOUN_PATTERNS)


def is_followup(text):
    """True if the text looks like a follow-up question (price of it, more details, etc.)."""
    return bool(_FOLLOWUP_HINTS.search(text))


def resolve_pronouns(text, session):
    """Substitute any pronoun in `text` with the session's remembered entity.

    Returns the resolved text. If the session has no remembered entity, or
    the text contains no pronouns, returns the text unchanged.
    """
    if not session or not session.get('last_entity'):
        return text
    entity = session['last_entity']
    resolved = text
    for pat in _PRONOUN_PATTERNS:
        resolved = pat.sub(entity, resolved)
    return resolved


def update_session(session, *, user_message, bot_response, intent, entity):
    """Append a turn and refresh the remembered entity."""
    if session is None:
        return
    session['history'].append({
        'user':   user_message,
        'bot':    bot_response[:300],
        'intent': intent,
        'entity': entity,
    })
    if len(session['history']) > MAX_HISTORY_TURNS:
        session['history'] = session['history'][-MAX_HISTORY_TURNS:]
    session['last_intent'] = intent
    if entity:
        session['last_entity'] = entity


def reset_session(session_id):
    """Drop a session - used by the 'Clear chat' UI button."""
    with _LOCK:
        _SESSIONS.pop(session_id, None)
