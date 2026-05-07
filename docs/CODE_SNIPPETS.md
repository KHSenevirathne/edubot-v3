# EduBot - Key Code Snippets and Source Layout

A short reference of the most important pieces of EduBot's code. Each
snippet has one or two lines explaining what it does and where it
lives in the repo.

---

## 13. Key Design Ideas with Code Snippets

### 13.1 Text Preprocessing

Normalises raw user text before classification: lower-case, strip
punctuation, lemmatise plurals/tenses, drop stop-words while keeping
question words. Source: [app/preprocess.py:97](app/preprocess.py).

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)            # drop punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    out = []
    for tok in tokens:
        tok = simple_lemmatize(tok)                 # 'courses' -> 'course'
        if tok in KEEP_WORDS or tok not in STOP_WORDS:
            out.append(tok)
    return ' '.join(out)
```

### 13.2 TF-IDF Vectorization

Turns each cleaned message into a 500-dimensional TF-IDF vector that
the classifier can read. Source: [app/train.py:93-97](app/train.py).

```python
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(patterns)              # shape (N_patterns, 500)
y = np.array(tags)                                  # parallel intent labels
```

### 13.3 SVM Model Training

Three classifiers are trained side-by-side; the one with the highest
5-fold cross-validation accuracy wins. Source: [app/train.py:108-128](app/train.py).

```python
models = {
    'Naive Bayes':   MultinomialNB(),
    'SVM':           SVC(kernel='linear', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    cv_scores = cross_val_score(model, X, y, cv=5)
    if cv_scores.mean() > best_cv:
        best_cv, best_name = cv_scores.mean(), name
```

### 13.4 Prediction and Confidence

Returns the predicted intent plus a 0–1 confidence from the SVM's
calibrated probability output. Source: [app/chat.py:159-170](app/chat.py).

```python
def predict_intent(self, user_input):
    cleaned = clean_text(user_input)
    vec = self.vectorizer.transform([cleaned])
    tag = self.model.predict(vec)[0]
    confidence = float(max(self.model.predict_proba(vec)[0]))
    return tag, confidence
```

### 13.5 Runtime Learning

When a user thumbs-downs a reply and suggests the right intent, the
suggestion lands in `learned_patterns` as **pending review** until an
admin approves it. Source: [app/learning.py:39-66](app/learning.py).

```python
def record_feedback(user_message, bot_response, predicted_intent,
                    confidence, helpful, expected_intent=None):
    db.log_feedback(user_message, bot_response, predicted_intent,
                    confidence, helpful, expected_intent)
    if not helpful and expected_intent:
        db.add_learned_pattern(
            pattern=user_message,
            intent=expected_intent,
            source='feedback_correction',
            approved=False,                # waits for admin review
        )
    return {'retrained': False, 'pending_review': db.count_pending_review()}
```

### 13.6 Retraining Logic

After every approved or admin-taught pattern, check the threshold
and retrain automatically once 5 approved-but-unused rows pile up.
Source: [app/learning.py:126-138](app/learning.py).

```python
AUTO_RETRAIN_THRESHOLD = 5

def _maybe_auto_retrain():
    if db.count_pending_patterns() >= AUTO_RETRAIN_THRESHOLD:
        _run_training()
        return True
    return False

def _run_training():
    from train import train_and_evaluate
    return train_and_evaluate(verbose=False)
```

### 13.7 Pronoun Resolution (multi-turn dialogue)

Substitutes anaphoric phrases like *it*, *this course*, *that one*
with the entity remembered from the previous turn so the bot can
answer follow-ups. Source: [app/context.py:110-122](app/context.py).

```python
def resolve_pronouns(text, session):
    if not session or not session.get('last_entity'):
        return text
    entity = session['last_entity']
    resolved = text
    for pat in _PRONOUN_PATTERNS:                   # 'it', 'this course', ...
        resolved = pat.sub(entity, resolved)
    return resolved
```

### 13.8 Emotional Intelligence Avatar

Picks an avatar mood (happy / neutral / confused / thinking) from
the bot's confidence + answer source so the user can see at a glance
how sure the bot is. Source: [static/script.js:592-607](static/script.js).

```javascript
function moodFor(confidence, source, tag) {
    if (tag === 'fallback' || source === 'fallback' || tag === 'error')
        return { name: 'confused', face: '\u{1F615}', label: "Hmm, I'm not sure" };
    if (tag === 'clarify')
        return { name: 'thinking', face: '\u{1F914}', label: 'Need a bit more info' };
    if (confidence >= 0.7 || source === 'database')
        return { name: 'happy',   face: '\u{1F642}', label: 'Happy to help' };
    if (confidence >= 0.4)
        return { name: 'neutral', face: '\u{1F610}', label: 'Best guess answer' };
    return     { name: 'confused', face: '\u{1F615}', label: 'Low confidence' };
}
```

### 13.9 Voice Recognition

Browser-native speech-to-text: when the mic button is pressed, the
Web Speech API transcribes the user's voice into the input box and
auto-sends the message. Source: [static/script.js:626-650](static/script.js).

```javascript
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = new SpeechRecognition();
recognition.lang = 'en-US';
recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    userInput.value = transcript;
    refreshSendButton();
    setTimeout(() => { if (!sendBtn.disabled) sendMessage(); }, 250);
};
```

### 13.10 Input Validation

Server-side gate that rejects gibberish, phone-number-style digit
runs and mostly-symbol input — letting emails, prices and years
through. Source: [app/validate.py:111-153](app/validate.py).

```python
def check_message_quality(text, field='message'):
    if _ALPHA_RUN_RE.search(text):                  # 30+ unbroken letters
        raise ValidationError(field,
            "that looks like gibberish - please ask a real question")
    if _DIGIT_RUN_RE.search(text):                  # 10+ unbroken digits
        raise ValidationError(field,
            "long number sequences aren't valid questions")
    if len(text) >= 4:
        letters = len(_LETTER_RE.findall(text))
        if letters / len(text) < MIN_LETTER_RATIO:  # < 30% letters
            raise ValidationError(field,
                "please ask a question in words")
```

### 13.11 Database Query

Parameterised SQL lookups for live answers — every dynamic intent
has its own helper. Free-text course search uses `LIKE` with safe
parameter binding. Source: [app/database.py:186-204](app/database.py).

```python
def list_courses():
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM courses ORDER BY level, name"
        )]

def find_course(keyword):
    pattern = f"%{keyword.lower()}%"
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(
            """SELECT * FROM courses
               WHERE LOWER(name)    LIKE ?
                  OR LOWER(code)    LIKE ?
                  OR LOWER(faculty) LIKE ?""",
            (pattern, pattern, pattern)
        )]
```

### 13.12 Admin Approval System

Admin endpoint that flips a pending pattern's `approved` flag to 1,
and may trigger an auto-retrain if the threshold is now met. The
bot is hot-swapped after a successful retrain. Source:
[app.py:241-252](app.py) and [app/learning.py:95-109](app/learning.py).

```python
# app.py
@app.route('/admin/approve/<int:pattern_id>', methods=['POST'])
@admin_required
def admin_approve(pattern_id):
    global bot
    result = learning.approve_suggestion(pattern_id)
    if result.get('retrained'):
        bot = EduBot()                              # hot-swap with new model
        result['model_reloaded'] = True
    return jsonify(result), 200 if result.get('approved') else 404

# app/learning.py
def approve_suggestion(pattern_id):
    if not db.approve_pattern(pattern_id):
        return {'approved': False, 'reason': 'not_found'}
    retrained = _maybe_auto_retrain()
    return {'approved': True, 'retrained': retrained}
```

---

## 14. Source Code Structure

### 14.1 Project Folder Structure

```
edubot-v3/
├── README.md                    quick start + feature overview
├── app.py                       Flask server + REST routes
├── build.spec                   PyInstaller config (single-file .exe)
├── requirements.txt             Python dependencies
├── render.yaml                  Render deployment config
├── Procfile                     gunicorn entry point
├── app/                         backend Python package
│   ├── chat.py                  inference engine + EduBot class
│   ├── context.py               sessions + anaphora resolution
│   ├── database.py              SQLite schema + read/write helpers
│   ├── learning.py              feedback-driven retraining loop
│   ├── preprocess.py            tokenise + lemmatise + stopwords
│   ├── seed_db.py               initial knowledge-base data
│   ├── train.py                 3-model training pipeline
│   └── validate.py              input validation + quality gate
├── data/
│   ├── intents.json             static patterns + small-talk responses
│   └── edubot.db                SQLite (gitignored, auto-seeded on boot)
├── models/
│   ├── chatbot_model.pkl        pickled SVM
│   ├── vectorizer.pkl           pickled TF-IDF vectoriser
│   └── model_info.txt           human-readable training metadata
├── templates/
│   ├── index.html               chat UI (mood avatar, mic, TTS)
│   └── admin.html               admin dashboard
├── static/
│   ├── script.js                chat UI + sessions + voice + toast
│   └── style.css                all styling
├── tests/
│   └── test_smoke.py            pytest smoke tests
└── docs/
    ├── TECHNICAL_DOCUMENTATION.md
    ├── DIAGRAMS.md              Mermaid source for all 10 diagrams
    └── CODE_SNIPPETS.md         this file
```

### 14.2 Main Application Files

The "engine room" — Python files in the `/app` package and `app.py`
at the project root.

| File | Role | LOC (approx) |
|---|---|---|
| `app.py` | Flask app, request routing, admin auth, hot-swap on retrain | ~310 |
| `app/chat.py` | `EduBot` class, intent prediction, keyword rescue, per-entity response builders | ~600 |
| `app/context.py` | Per-session memory, anaphora resolution, entity tracking | ~150 |
| `app/preprocess.py` | `clean_text()`, `simple_lemmatize()`, stopword whitelist | ~140 |
| `app/train.py` | 3-model bake-off, 5-fold CV, pickle output | ~210 |
| `app/learning.py` | Feedback recording, admin approve/discard, retrain trigger | ~150 |
| `app/validate.py` | `clean_text_field`, `check_message_quality`, `ValidationError` | ~200 |
| `app/database.py` | SQLite schema, all read/write helpers, idempotent migrations | ~415 |
| `app/seed_db.py` | Seed data for the 7 read-only knowledge tables | ~210 |

### 14.3 Frontend Files

The user-facing layer.

| File | Role |
|---|---|
| `templates/index.html` | Chat UI markup: sidebar, mood avatar, mic button, TTS toggle, input bar |
| `templates/admin.html` | Admin dashboard: stats grid, pending review table, teach form, retrain button |
| `static/script.js` | Chat send/receive, session id, mood avatar, voice I/O, toast, client-side validation |
| `static/style.css` | All styling: layout, chat bubbles, mood transitions, toast popup |

### 14.4 Database Files

The persistence layer.

| File | Role |
|---|---|
| `data/intents.json` | 15 intent tags, ~999 training patterns + static small-talk responses |
| `data/edubot.db` | SQLite file with 10 tables (gitignored; `app.py` auto-seeds it on first boot if empty) |
| `app/database.py` | Schema definition (`init_schema()`) + every read/write helper used by the rest of the app |
| `app/seed_db.py` | One-shot seed script: `python app/seed_db.py` populates the 7 read-only tables; preserves feedback / learned_patterns / chat_history |

### 14.5 Testing Files

| File | Role |
|---|---|
| `tests/test_smoke.py` | 25 pytest tests covering preprocessing, DB seeding, intent classification, the three-tier response path, the ML feedback loop, and input validation |

Run them with:

```bash
python -m pytest tests/ -q
# expected: 25 passed in ~5 s
```
