"""
app.py - Flask Web Server for EduBot v3

Wires the natural-language interface (templates + static) to the
inference engine (chat.EduBot) and the learning loop (learning.py).

Endpoints:
  GET  /             chat UI
  POST /chat         classify message, return DB- or template-based answer
  POST /feedback     persist thumbs-up/down (drives the ML loop)
  POST /teach        admin/user explicitly maps pattern -> intent
  POST /retrain      force a model retrain
  GET  /admin        admin dashboard (DB stats, recent feedback)
  GET  /api/intents  list available intents (used by the teach modal)
  GET  /api/stats    JSON stats endpoint (used in tests)
  GET  /health       liveness probe
"""

import os
import sys
import json
from functools import wraps

from flask import Flask, Response, request, jsonify, render_template
from flask_cors import CORS

# Make the /app package importable regardless of CWD.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app'))

from chat import EduBot              # noqa: E402
import learning                       # noqa: E402
import database as db                 # noqa: E402
import validate as v                  # noqa: E402
import context as ctx                 # noqa: E402


app = Flask(__name__, template_folder='templates', static_folder='static')
# Cap incoming JSON body size as a defence against junk payloads.
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024     # 64 KB
CORS(app)


@app.errorhandler(v.ValidationError)
def _handle_validation_error(err):
    """Turn ValidationError into a clean 400 response."""
    return jsonify({'error': err.message, 'field': err.field}), 400


# ---------------- Admin auth ----------------
# Set EDUBOT_ADMIN_PASSWORD in the environment to require Basic Auth on
# /admin and any endpoint that mutates the model. If unset (typical
# during local development), admin endpoints are open and we log a
# warning at startup.

ADMIN_PASSWORD = os.environ.get('EDUBOT_ADMIN_PASSWORD')
if not ADMIN_PASSWORD:
    print("[admin] EDUBOT_ADMIN_PASSWORD not set - /admin is OPEN. "
          "Set the env var before deploying publicly.")


def admin_required(view):
    """Decorator: require Basic Auth on the wrapped view if a password
    is configured. Username is ignored - we only check the password.
    """
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not ADMIN_PASSWORD:
            return view(*args, **kwargs)
        auth = request.authorization
        if auth and auth.password == ADMIN_PASSWORD:
            return view(*args, **kwargs)
        return Response(
            'Authentication required.', 401,
            {'WWW-Authenticate': 'Basic realm="EduBot Admin"'},
        )
    return wrapper

# Cold start: prepare DB schema, then load the trained model.
db.init_schema()

# Auto-seed on a fresh deploy. The DB is intentionally NOT committed
# to git (the bot writes to it at runtime, so tracking the file
# causes merge conflicts on every redeploy). Instead, when the
# server boots against an empty courses table we populate the seed
# tables ourselves. Idempotent on subsequent boots.
if db.stats()['courses'] == 0:
    print("[boot] empty knowledge base detected - running seed_db.seed_all()")
    import seed_db                              # noqa: E402
    seed_db.seed_all()

print("Loading EduBot model...")
bot = EduBot()
print("EduBot ready.")


# ---------------- Routes ----------------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Classify a user message and return the bot's answer.

    Optional `session_id` in the body wires the request into the
    multi-turn dialogue manager so the bot can resolve pronouns
    ("price of it") to the previously discussed entity.
    """
    data = request.get_json(silent=True) or {}
    user_message = v.clean_text_field(
        data.get('message'), 'message',
        min_len=1, max_len=v.MAX_MESSAGE_LEN,
        check_quality=True,     # gibberish / phone-number gate
    )

    # Session ID is opaque - we only validate length and that it looks
    # like a hex token, since the frontend mints it via crypto.randomUUID.
    raw_sid = data.get('session_id')
    session_id = None
    if isinstance(raw_sid, str) and 0 < len(raw_sid) <= 64:
        if all(c.isalnum() or c == '-' for c in raw_sid):
            session_id = raw_sid
    session = ctx.get_session(session_id) if session_id else None

    result = bot.get_response(user_message, session=session)
    return jsonify({
        'response':   result['response'],
        'tag':        result['tag'],
        'confidence': result['confidence'],
        'source':     result['source'],
        'entity':     result.get('entity'),
        'session_id': session['id'] if session else None,
    })


@app.route('/session/reset', methods=['POST'])
def session_reset():
    """Clear server-side memory for a session (paired with the
    'Clear chat' button so a new conversation starts cleanly)."""
    data = request.get_json(silent=True) or {}
    raw_sid = data.get('session_id')
    if isinstance(raw_sid, str) and raw_sid:
        ctx.reset_session(raw_sid)
    return jsonify({'reset': True})


@app.route('/feedback', methods=['POST'])
def feedback():
    """Persist a thumbs-up/thumbs-down and possibly trigger a retrain.

    Expected body:
      { "user_message", "bot_response", "predicted_intent",
        "confidence", "helpful": bool, "expected_intent"? }
    """
    global bot
    data = request.get_json(silent=True) or {}

    user_message = v.clean_text_field(
        data.get('user_message'), 'user_message',
        max_len=v.MAX_MESSAGE_LEN,
    )
    # bot_response can be long (DB-backed answers list every course).
    bot_response = v.clean_text_field(
        data.get('bot_response') or '-', 'bot_response',
        min_len=1, max_len=4000,
    )
    helpful = v.parse_bool(data.get('helpful'), 'helpful')
    confidence = v.parse_float_in_range(
        data.get('confidence'), 'confidence', lo=0.0, hi=1.0,
    )

    predicted_intent = data.get('predicted_intent')
    if predicted_intent is not None:
        predicted_intent = v.clean_text_field(
            predicted_intent, 'predicted_intent', max_len=64,
        )

    expected_intent = data.get('expected_intent')
    if expected_intent:
        expected_intent = v.validate_intent(
            expected_intent, 'expected_intent',
            allowed=set(bot.response_map.keys()),
        )

    result = learning.record_feedback(
        user_message=user_message,
        bot_response=bot_response,
        predicted_intent=predicted_intent,
        confidence=confidence,
        helpful=helpful,
        expected_intent=expected_intent,
    )

    if result['retrained']:
        bot = EduBot()
        result['model_reloaded'] = True

    return jsonify(result)


@app.route('/teach', methods=['POST'])
@admin_required
def teach():
    """Admin-only: directly add an APPROVED (pattern, intent) row.

    Expected body: { "pattern": str, "intent": str }
    """
    global bot
    data = request.get_json(silent=True) or {}

    # Patterns are short - reject anything that looks like an essay.
    pattern = v.clean_text_field(
        data.get('pattern'), 'pattern',
        min_len=2, max_len=v.MAX_PATTERN_LEN,
    )
    intent = v.validate_intent(
        data.get('intent'), 'intent',
        allowed=set(bot.response_map.keys()),
    )

    result = learning.teach(pattern, intent)
    if result['retrained']:
        bot = EduBot()
        result['model_reloaded'] = True

    return jsonify(result)


@app.route('/retrain', methods=['POST'])
@admin_required
def retrain():
    """Admin-only: force a retrain. Returns the new model's name."""
    global bot
    model_name = learning.manual_retrain()
    bot = EduBot()
    return jsonify({'retrained': True, 'model': model_name})


@app.route('/admin/approve/<int:pattern_id>', methods=['POST'])
@admin_required
def admin_approve(pattern_id):
    """Admin curates: approve a pending suggestion so it enters training."""
    global bot
    result = learning.approve_suggestion(pattern_id)
    if result.get('retrained'):
        bot = EduBot()
        result['model_reloaded'] = True
    status = 200 if result.get('approved') else 404
    return jsonify(result), status


@app.route('/admin/discard/<int:pattern_id>', methods=['POST'])
@admin_required
def admin_discard(pattern_id):
    """Admin curates: discard a pending suggestion (deletes the row)."""
    result = learning.discard_suggestion(pattern_id)
    status = 200 if result['discarded'] else 404
    return jsonify(result), status


@app.route('/api/intents', methods=['GET'])
def list_intents():
    """List intent tags + a sample response. Used by the teach modal."""
    return jsonify({
        'intents': [
            {
                'tag': intent['tag'],
                'sample': intent['responses'][0] if intent['responses'] else '',
            }
            for intent in bot.intents_data['intents']
            if intent['tag'] != 'fallback'
        ]
    })


@app.route('/api/stats', methods=['GET'])
def api_stats():
    """JSON stats. Used by tests/test_endpoints.py and the admin page."""
    return jsonify(db.stats())


@app.route('/admin', methods=['GET'])
@admin_required
def admin():
    """Admin page - DB stats, pending suggestions, recent feedback,
    teach form, retrain button."""
    with db.get_connection() as conn:
        recent = [dict(r) for r in conn.execute(
            "SELECT * FROM feedback ORDER BY id DESC LIMIT 25"
        )]
        learned = [dict(r) for r in conn.execute(
            "SELECT * FROM learned_patterns "
            "WHERE approved = 1 ORDER BY id DESC LIMIT 25"
        )]
    pending = db.get_pending_patterns()
    return render_template(
        'admin.html',
        stats=db.stats(),
        feedback_rows=recent,
        learned_rows=learned,
        pending_rows=pending,
        intents=sorted(bot.response_map.keys()),
    )


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'bot': 'EduBot v3 is running'})


if __name__ == '__main__':
    # Local development entry point. In production (Render, etc.) the
    # service starts gunicorn directly against the `app` object above
    # via the Procfile, so this block is bypassed.
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug)
