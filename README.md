# EduBot v3 - University Support Chatbot

An NLP-driven university help-desk chatbot built from scratch (no third-party
LLM APIs). v3 closes the gaps in v2 against the assignment brief and adds
multi-turn dialogue, an emotional-feedback avatar, and voice I/O:

- **Three-tier architecture** - natural-language interface, inference engine,
  and a SQLite **knowledge base** for facts that change over time.
- **Multi-turn dialogue** - per-session memory resolves "it" / "this course"
  to the previously discussed programme, so a real consulting flow works
  ("show me courses" -> "tell me about MBA" -> "price of it").
- **Machine-learning loop with trust gate** - users flag wrong answers via
  thumbs-down; suggestions land in an admin review queue; approved patterns
  trigger an auto-retrain.
- **Keyword-rescue safety net** - when the classifier returns sub-threshold
  confidence (<0.4), the message is scanned for an unambiguous intent
  keyword (event, hostel, scholarship, lecturer, ...) before falling back,
  so close-to-the-line questions still land on the right answer.
- **Emotional intelligence** - mood-changing avatar (happy / neutral /
  confused / thinking) reflects the bot's confidence in the last reply.
- **Voice I/O** - Web Speech API microphone for spoken questions, optional
  read-aloud of the bot's replies via `speechSynthesis`.
- **Input quality gate** - server- and client-side checks reject keyboard
  mashing and phone-number-style digit runs while letting emails, prices
  and years through.
- **Single-file executable** - PyInstaller spec ships with the project so
  the application runs "without extra installation of libraries."

## Tech stack

| Layer | Tech |
|---|---|
| NLP   | scikit-learn (TF-IDF + Naive Bayes / Linear SVM / Random Forest), custom lemmatizer |
| Dialogue | In-memory session store with anaphora resolution and entity tracking |
| Web   | Flask, Flask-CORS |
| Data  | SQLite (stdlib `sqlite3`), JSON for static intents |
| UI    | HTML / CSS / vanilla JS, Web Speech API, speechSynthesis |
| Build | PyInstaller |

## Quick start

```bash
# from the project root
python -m venv venv
venv\Scripts\activate         # Windows

pip install -r requirements.txt

# 1) train the classifier (model files ARE in git; only re-run after
#    you change intents.json or merge in approved learned patterns)
python app/train.py

# 2) run the web app - auto-seeds the SQLite DB on first boot
python app.py
# open http://localhost:5000
```

> The SQLite DB (`data/edubot.db`) is **not** committed to git.
> `app.py` runs `seed_db.seed_all()` automatically the first time it
> sees an empty `courses` table, then leaves the live DB alone on
> every subsequent boot. To re-seed manually (refreshes courses,
> events, faculty, etc. without touching feedback / chat_history /
> learned_patterns) run `python app/seed_db.py`.

Optional: set `EDUBOT_ADMIN_PASSWORD=...` before starting `app.py` to require
Basic Auth on `/admin`, `/teach` and `/retrain`. With it unset, the admin
dashboard is open (development mode).

## Build a standalone .exe

```bash
pip install pyinstaller==6.10.0
pyinstaller build.spec --noconfirm
# result: dist/EduBot.exe
```

## Project layout

```
edubot-v3/
├── app.py                    Flask server + REST endpoints
├── app/
│   ├── preprocess.py         Tokenisation, lemmatisation, stopword removal
│   ├── database.py           SQLite schema + read/write helpers
│   ├── seed_db.py            Initial knowledge-base data
│   ├── train.py              3-model training pipeline (NB / SVM / RF)
│   ├── chat.py               Inference engine + DB-backed response builder
│   ├── context.py            Multi-turn dialogue: sessions + anaphora
│   ├── validate.py           Input validation + message quality gate
│   └── learning.py           Feedback-driven retraining loop
├── data/
│   ├── intents.json          Static patterns + small-talk responses
│   └── edubot.db             SQLite (auto-created)
├── models/                   Pickled model + vectorizer (auto-created)
├── templates/
│   ├── index.html            Chat UI
│   └── admin.html            Admin dashboard
├── static/                   CSS / JS assets (mood avatar, voice I/O)
├── tests/                    Pytest smoke tests
├── docs/                     Technical documentation + design notes
├── build.spec                PyInstaller config
└── requirements.txt
```

## API endpoints

| Method | Path                | Purpose |
|--------|---------------------|---------|
| GET    | `/`                 | Chat UI |
| POST   | `/chat`             | Predict intent + return DB- or template-based answer; accepts optional `session_id` for multi-turn |
| POST   | `/session/reset`    | Clear server-side memory for a session (paired with the Clear-chat button) |
| POST   | `/feedback`         | Record thumbs-up/down, optionally suggest correct intent |
| POST   | `/teach`            | Admin: add `(pattern, intent)` pre-approved |
| POST   | `/retrain`          | Admin: force a model retrain |
| POST   | `/admin/approve/<id>` | Admin: approve a pending pattern |
| POST   | `/admin/discard/<id>` | Admin: discard a pending pattern |
| GET    | `/admin`            | Admin dashboard |
| GET    | `/api/intents`      | List intent tags |
| GET    | `/api/stats`        | DB row counts |
| GET    | `/health`           | Liveness probe |

## Multi-turn dialogue

The bot keeps a small per-session memory keyed by an opaque session id
that the browser mints once and stores in `localStorage`. On every turn
the dialogue manager:

1. Looks for **pronouns** (`it`, `this course`, `that one`, `the
   programme`) and substitutes them with the entity remembered from the
   previous turn.
2. Extracts a **course entity** from the message (canonical name,
   course code, or alias such as `cs`, `mba`, `data science`).
3. If the user used a pronoun before any course was in context, the bot
   asks **"Which programme would you like to know about?"** instead of
   guessing.
4. Routes per-course questions to dedicated response builders, e.g.
   `_respond_fees_for_course(name)` returns just the fee for the course
   in context (per-year, per-semester and total).

Sessions live in `app/context.py`, expire after 1 hour idle, and are
capped at 1000 active sessions.

Example flow:

```
USER> What courses do you offer?
BOT > [lists all programmes]
USER> Tell me about MBA
BOT > [MBA detail card; entity = MBA]
USER> price of it
BOT > [MBA fees, $5000/year, $2500/semester, ~$7500 total]
USER> how do I apply for it
BOT > [admission steps; entity = MBA still]
```

## Mood avatar (emotional intelligence)

The header avatar swaps face + colour depending on the bot's confidence
in its last answer:

| Mood       | When                                                           |
|------------|----------------------------------------------------------------|
| 🙂 Happy   | confidence >= 0.7 OR answer came from the database             |
| 😐 Neutral | confidence between 0.4 and 0.7                                 |
| 😕 Confused | fallback / error / very low confidence                        |
| 🤔 Thinking | shown while the request is in flight                          |

Each bot message bubble also gets a coloured avatar matching its own
confidence, so historical context is preserved as you scroll back.

## Voice I/O

- **Microphone button** in the input bar uses `webkitSpeechRecognition`
  (Chrome / Edge) to transcribe spoken questions and auto-sends them.
  Browsers without the API see a disabled button.
- **Speaker button** in the header toggles read-aloud via
  `speechSynthesis`. The preference is persisted in `localStorage`. Bot
  replies are stripped of bullet markup before being spoken.

## Input validation

Two layers, kept in sync:

- **Client side** (`static/script.js`): rejects gibberish (alpha runs >
  30 chars), phone-number-style digit runs (> 15 digits), and
  mostly-symbol input (letter ratio < 30%). The send button stays
  disabled and a tooltip shows the reason.
- **Server side** (`app/validate.py:check_message_quality`): same three
  rules enforced again so the API can't be bypassed. Validation errors
  return HTTP 400 with `{ error, field }` and are surfaced inline in
  the chat.

Emails, URLs, prices like `$3000` and years like `2024-2025` all pass.

## ML feedback loop (two-tier trust)

```
user message -> /chat -> bot reply (with confidence + source badge)
                             |
                             +-- 👍/👎 buttons under every reply
                             |
                             +-- 👎 + correct intent
                                            |
                                            v
                                  learned_patterns
                                  (approved = 0)         <-- pending
                                            |
                                  Admin reviews on /admin
                                            |
                              Approve   |   Discard (delete)
                                  v
                          approved = 1
                                            |
              every 5 approved-but-unused patterns OR
              admin clicks Retrain
                                            |
                                            v
                                    train.py merges
                              intents.json + approved learned_patterns
                                            |
                                            v
                                   new chatbot_model.pkl
                                   (Flask hot-swaps it)
```

End users **cannot** poison the model directly - their suggestions stay
behind the admin gate.

## Further reading

| File                              | What's in it                                  |
|-----------------------------------|-----------------------------------------------|
| `docs/TECHNICAL_DOCUMENTATION.md` | Full system design, architecture, rationale  |
| `docs/PSEUDOCODE.md`              | Pseudocode for the main algorithms           |
| `docs/DIAGRAMS.md`                | Architecture, ER, and dialogue-flow diagrams |
| `docs/CODE_SNIPPETS.md`           | Annotated highlights of the key modules      |
