# EduBot v3 - University Support Chatbot

An NLP-driven university help-desk chatbot built from scratch (no third-party
LLM APIs). v3 closes the gaps in v2 against the assignment brief:

- **Three-tier architecture** - natural-language interface, inference engine,
  and a SQLite **knowledge base** for facts that change over time.
- **Machine-learning loop** - users can flag wrong answers and teach the bot
  the correct intent. Patterns persist across sessions and the bot retrains.
- **Single-file executable** - PyInstaller spec ships with the project so the
  application runs "without extra installation of libraries."

## Tech stack

| Layer | Tech |
|---|---|
| NLP   | scikit-learn (TF-IDF + Naive Bayes / Linear SVM / Random Forest), custom lemmatizer |
| Web   | Flask, Flask-CORS |
| Data  | SQLite (stdlib `sqlite3`), JSON for static intents |
| UI    | HTML / CSS / vanilla JS |
| Build | PyInstaller |

## Quick start

```bash
# from the project root
python -m venv venv
venv\Scripts\activate         # Windows
pip install -r requirements.txt

# 1) seed the knowledge base
python app/seed_db.py

# 2) train the classifier
python app/train.py

# 3) run the web app
python app.py
# open http://localhost:5000
```

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
│   └── learning.py           Feedback-driven retraining loop
├── data/
│   ├── intents.json          Static patterns + small-talk responses
│   └── edubot.db             SQLite (auto-created)
├── models/                   Pickled model + vectorizer (auto-created)
├── templates/
│   ├── index.html            Chat UI
│   └── admin.html            Admin dashboard
├── static/                   CSS / JS assets
├── tests/                    Pytest smoke tests
├── docs/                     Technical documentation + design notes
├── build.spec                PyInstaller config
└── requirements.txt
```

## API endpoints

| Method | Path           | Purpose |
|--------|----------------|---------|
| GET    | `/`            | Chat UI |
| POST   | `/chat`        | Predict intent, return DB- or template-based answer |
| POST   | `/feedback`    | Record thumbs-up/down, optionally trigger retrain |
| POST   | `/teach`       | Add `(pattern, intent)` to the knowledge base |
| POST   | `/retrain`     | Force a model retrain |
| GET    | `/admin`       | Admin dashboard |
| GET    | `/api/intents` | List intent tags |
| GET    | `/api/stats`   | DB row counts |
| GET    | `/health`      | Liveness probe |

## ML feedback loop

```
user message -> /chat -> bot reply (with confidence + source badge)
                             |
                             +-- 👍/👎 buttons under every reply
                             |
                             +-- 👎 + correct intent -> /feedback
                                            |
                                            v
                              learned_patterns table
                                            |
              every 5 new patterns OR /retrain triggers
                                            |
                                            v
                                    train.py merges
                              intents.json + learned_patterns
                                            |
                                            v
                                   new chatbot_model.pkl
```

The full design is documented in `docs/TECHNICAL_DOCUMENTATION.md`.
