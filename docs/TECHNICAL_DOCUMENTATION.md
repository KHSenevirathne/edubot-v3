# EduBot v3 - Technical Documentation

**Module:** Artificial Intelligence
**Assignment:** 2 (75% - AI artifact)
**Option chosen:** 2 - Virtual Assistant (Chat Bot)
**Domain:** Education counselling / university student support

---

## Table of contents

1. [Product description](#1-product-description)
2. [Research and challenges](#2-research-and-challenges)
3. [Design architecture](#3-design-architecture)
4. [P.E.A.S. analysis](#4-peas-analysis)
5. [AI traits found in the product](#5-ai-traits-found-in-the-product)
6. [Algorithms (flowcharts and pseudocode)](#6-algorithms-flowcharts-and-pseudocode)
7. [Key design ideas with code snippets](#7-key-design-ideas-with-code-snippets)
8. [UML diagrams](#8-uml-diagrams)
9. [Source code listing](#9-source-code-listing)
10. [Test plan, test data and results](#10-test-plan-test-data-and-results)
11. [Conclusion](#11-conclusion)
12. [References](#12-references)

---

## 1. Product description

EduBot is a text-based conversational assistant that answers prospective and
current student questions about a (fictional) university. It accepts natural
language input through a web interface, classifies the user's *intent* using
machine learning, and replies with information drawn from either a SQLite
knowledge base (for facts that change over time, such as fees and exam
dates) or a static template store (for small-talk such as greetings).

The system was built from scratch using classical NLP and machine learning
techniques (no third-party LLM APIs). It demonstrates the canonical
three-tier chatbot architecture required by the brief, plus a feedback-driven
machine-learning loop that lets the chatbot **update its own knowledge base**
when it gets something wrong.

### Key features

- Recognises 14 distinct intents across the university support domain
  (courses, fees, admission, scholarships, exams, timetable, library,
  contact, faculty, hostel, events, plus greeting / goodbye / thanks).
- Sub-second response time on a laptop CPU (no GPU required).
- Live data answers - the response for *"what are the fees?"* is built
  by querying SQLite, not by reading a hard-coded JSON file.
- User-driven learning - thumbs-down feedback collects new training
  patterns and a retrain rebuilds the model with them merged in.
- Confidence-aware fallback - low-confidence predictions (< 0.4) are
  routed to a polite "I didn't understand" response instead of
  guessing.
- Single-binary distribution via PyInstaller.

### How a user interacts with it

A student opens `http://localhost:5000`, sees a chat window with eight
quick-action buttons, types or clicks a question, and receives a
formatted answer plus a confidence badge and a source badge
(`database` / `static` / `fallback`). Below each bot answer are
thumbs-up and thumbs-down buttons. A thumbs-down opens a small picker
to select the intent the bot *should* have used; submitting it appends
the (phrase, intent) pair to a `learned_patterns` table. Once five
patterns have accumulated, or when an admin clicks "Retrain model
now" on the `/admin` page, the classifier is rebuilt with the new
patterns merged in.

---

## 2. Research and challenges

### Background reading

| Topic | Source |
|---|---|
| Bag-of-words and TF-IDF for short-text classification | Manning et al., *Introduction to Information Retrieval* (Cambridge, 2008), ch. 6 |
| Multinomial Naive Bayes for text | scikit-learn user guide, "Naive Bayes" |
| Linear SVMs for sparse high-dimensional features | Joachims (1998), *Text Categorization with Support Vector Machines* |
| Random Forests | Breiman (2001), *Random Forests* (Machine Learning 45) |
| Stratified k-fold cross-validation | Kohavi (1995), IJCAI |
| Three-tier chatbot pattern (UI / inference / KB) | Jurafsky & Martin, *Speech and Language Processing*, ch. 24 |
| SQLite design rationale | Hipp, *Architecture of SQLite* (sqlite.org) |
| Flask routing patterns | Grinberg, *Flask Web Development* (O'Reilly, 2018) |

### Design questions investigated

1. **Which classifier should we use?**
   We trained Multinomial Naive Bayes, a Linear SVM and a Random
   Forest, then picked the winner by 5-fold stratified
   cross-validation. Naive Bayes is the textbook short-text baseline,
   linear SVM tends to be best on sparse high-dimensional TF-IDF
   features (per Joachims 1998), and Random Forest is a strong
   non-parametric baseline. SVM won on our corpus
   ([see results](#training-output)).

2. **How should the knowledge base be split between intents.json and the database?**
   The brief explicitly says *"static facts may be hard coded ... but the
   facts that can change/update with time must be obtained from a
   database."* We therefore split intents into:

   - **Static** (`greeting`, `goodbye`, `thanks`, `fallback`) - response
     templates live in `intents.json`, picked at random.
   - **Dynamic** (everything else: courses, fees, faculty, events,
     etc.) - each one has a Python builder that issues SQL queries
     and formats the result.

   The `DYNAMIC_INTENTS` set in `app/chat.py` is the single source of
   truth for which is which.

3. **How does the bot "learn" without retraining on every message?**
   Online learning is brittle and would let a single malicious user
   poison the model. Instead, EduBot uses a **batch-correction**
   approach: feedback writes to a `learned_patterns` table, and a
   threshold (default 5 pending patterns) or a manual `/retrain`
   triggers a full retrain that merges those rows into the corpus.

### Challenges encountered and how they were solved

| Challenge | Resolution |
|---|---|
| `predict_proba` on `SVC(kernel='linear')` is slow and uses Platt scaling, which can disagree with the predicted class label. | Accepted the small inconsistency; users care about the *answer* more than the *probability calibration*. We use the probability only to drive the fallback threshold, not to rank intents. |
| Confidence threshold of 0.4 was too aggressive on greeting + small-talk variants. | Kept the threshold at 0.4 *but* enriched the greeting/goodbye/thanks pattern lists in `intents.json` (60+ phrases each, including typos and full sentences). |
| Retraining inside a Flask request handler blocks the thread for ~3 seconds. | Acceptable for a single-user demo. For production this would move to a background `concurrent.futures.ThreadPoolExecutor` or a dedicated worker. Documented as future work. |
| Lemmatisation usually requires the `nltk.WordNetLemmatizer`, which downloads ~30 MB of corpus data on first use. | Replaced with a hand-built lookup table + suffix-stripping rules in `app/preprocess.py`. Slightly less accurate but ships inside the PyInstaller bundle and works offline. |
| The bot would not "self-update" without a way for the user to tell it the right answer. | Added a two-tier loop: end users flag with thumbs-up / thumbs-down (`/feedback`) and *suggest* an intent; an admin approves or discards each suggestion on `/admin` before it can enter training. The `/teach` form is admin-only. |
| Naive design would let any visitor poison the model. | Split the learning loop into Tier 1 (open-feedback, approved=0 by default) and Tier 2 (admin-curated, approved=1). `train.py` only consumes approved rows. `/admin*` and `/teach`, `/retrain` are guarded by `@admin_required` Basic Auth (env var `EDUBOT_ADMIN_PASSWORD`). |
| Re-seeding the DB during development would also wipe taught patterns. | `seed_db.py` only `DELETE`s the seven seed tables; `feedback`, `learned_patterns` and `chat_history` survive. |

---

## 3. Design architecture

### Three-tier architecture (per the brief)

```
+---------------------------------------------------------------+
|                  Tier 1 - Natural Language                    |
|                          Interface                            |
|                                                               |
|   templates/index.html  +  static/script.js  +  style.css     |
|   - chat window + quick actions                               |
|   - feedback buttons (drives ML loop)                         |
|   - admin dashboard (/admin) - Teach form, pending review, retrain |
|   - admin dashboard (/admin)                                  |
+---------------------------+-----------------------------------+
                            |
                            |  HTTP/JSON
                            v
+---------------------------------------------------------------+
|                  Tier 2 - Inference Engine                    |
|                                                               |
|     app.py (routes: /chat /feedback /teach /retrain           |
|             /admin /admin/approve/<id> /admin/discard/<id>)   |
|     app/chat.py  (EduBot class - intent classification)       |
|     app/preprocess.py  (clean_text: tokenise + lemmatise)     |
|     app/train.py  (NB / SVM / RF + 5-fold CV)                 |
|     app/learning.py  (feedback -> learned_patterns -> retrain)|
+---------------------------+-----------------------------------+
                            |
                            |  function calls
                            v
+---------------------------------------------------------------+
|                Tier 3 - Knowledge Base / DB                   |
|                                                               |
|   data/intents.json  (static patterns + small-talk responses) |
|                                                               |
|   data/edubot.db (SQLite)                                     |
|   - courses, faculty, events, exams, scholarships             |
|   - hostel_rooms, kv_facts (library/contact/timetable)        |
|   - feedback, learned_patterns, chat_history                  |
+---------------------------------------------------------------+
```

### Component-level data flow for one chat turn

```
  User types message
         |
         v
  POST /chat  (Flask)
         |
         v
  EduBot.get_response(text)
         |
         +--> preprocess.clean_text(text)        # lower-case, strip,
         |       returns cleaned tokens          # lemmatise
         |
         +--> vectorizer.transform(...)          # TF-IDF
         |
         +--> model.predict + predict_proba      # SVM (best of 3)
         |       returns (intent_tag, conf)
         |
         +--> if conf < 0.4: tag = 'fallback'
         |
         +--> if tag in DYNAMIC_INTENTS:
         |        response = builder(tag)        # query SQLite
         |        source = 'database'
         |    else:
         |        response = random.choice(...)  # intents.json
         |        source = 'static'
         |
         +--> db.log_chat(...)                   # for analytics
         |
         v
  JSON response back to browser
         |
         v
  Browser renders bubble + 👍/👎 buttons
```

---

## 4. P.E.A.S. analysis

(Performance / Environment / Actuators / Sensors - the standard
intelligent-agent specification.)

| Element     | Description |
|-------------|-------------|
| **Performance** | Intent-classification accuracy on held-out data (84% on the v3 corpus); 5-fold cross-validation accuracy (78%); proportion of queries resolved without falling back; user satisfaction expressed via 👍/👎; reply latency (<1 s on a laptop CPU); proportion of dynamic-intent answers correctly sourced from the database. |
| **Environment** | A web browser. The user is a prospective or current student typing university questions in English. The environment is fully observable (one message in, one message out), single-agent (one user at a time per session), discrete (text), partly stochastic (the random response selector for static intents), and dynamic from the bot's perspective (its own knowledge base updates over time as users teach it). |
| **Actuators** | Outbound text - rendered as a chat bubble in the user's browser. The bot also writes to its own SQLite database (chat history, feedback, learned patterns) and triggers retrains, so it acts on its own knowledge base. |
| **Sensors** | Inbound text - the JSON `message` field on `POST /chat`. Indirectly the bot also "senses" user satisfaction through the `/feedback` endpoint. |

---

## 5. AI traits found in the product

The brief lists six possible AI traits and asks the team to demonstrate
several. EduBot v3 deliberately exhibits **three**:

### Trait 1 - Natural Language Processing (primary)

User input is parsed by a hand-written NLP pipeline:

1. Lower-casing.
2. Punctuation stripping with a regex.
3. Whitespace tokenisation.
4. Lemmatisation - irregular plurals (`courses` → `course`,
   `faculties` → `faculty`) and common verb conjugations
   (`applying`/`applied`/`applies` → `apply`) via a lookup table,
   plus suffix-stripping (`-ing`, `-ed`, `-ies`, `-s`) for words not
   in the table.
5. Stopword removal (with a whitelist that keeps interrogatives like
   *what*, *how*, *when*).
6. TF-IDF vectorisation into a 500-feature sparse vector.
7. Multi-class classification with a Linear SVM, returning a tag and
   a probability.

This is the bulk of the 22-mark "effective implementation of NLP via
a program/inference engine" line in the marking scheme.

### Trait 2 - Decision making / inference

Once the intent is predicted, the bot decides:

- whether the prediction is confident enough to use (threshold = 0.4);
- whether the response should be drawn from the live database or
  from the static template file;
- when to trigger an automatic retrain (5+ pending learned patterns).

### Trait 3 - Learning

The bot learns in two ways:

1. **At training time** - it picks the best of three classifiers
   (Naive Bayes vs SVM vs Random Forest) by cross-validation
   accuracy.
2. **At runtime** - the bot learns through a two-tier feedback loop.
   End users flag wrong answers with 👎 and may suggest the correct
   intent; the suggestion is queued for admin review. Admins approve
   or discard suggestions on the `/admin` dashboard, and may also
   directly teach (pre-approved) patterns. Approved patterns are
   merged into the next training run, after which the bot
   demonstrably handles phrasings it could not before. This
   satisfies the marking-scheme line *"an indication of machine
   learning - chatbot updating its own knowledge base"* while
   protecting the model from data-poisoning by untrusted users.

---

## 6. Algorithms (flowcharts and pseudocode)

### Algorithm A - Intent classification + response generation

```
ALGORITHM get_response(user_input):
    if user_input is empty:
        return "Please type a question..."

    cleaned   <- clean_text(user_input)
    vector    <- tfidf_vectorizer.transform(cleaned)
    tag, conf <- model.predict(vector), max(model.predict_proba(vector))

    if conf < CONFIDENCE_THRESHOLD (0.4):
        tag <- "fallback"

    if tag in DYNAMIC_INTENTS:
        response <- build_database_response(tag)
        source   <- "database"
        if response is None:
            response <- random_static_template(tag)
            source   <- "static"
    else:
        response <- random_static_template(tag)
        source   <- "static"

    log_chat_history(user_input, response, tag, conf, source)
    return { tag, response, conf, source }
```

### Algorithm B - Text preprocessing

```
ALGORITHM clean_text(text):
    text <- lowercase(text)
    text <- regex_replace(text, "[^a-z\s]", "")    # drop punctuation
    text <- regex_replace(text, "\s+", " ")        # collapse spaces
    tokens <- split(text, " ")

    output <- []
    for token in tokens:
        token <- simple_lemmatize(token)
        if token not in STOP_WORDS or token in KEEP_WORDS:
            output.append(token)

    return join(output, " ")


ALGORITHM simple_lemmatize(word):
    if word in LEMMA_RULES:
        return LEMMA_RULES[word]               # explicit dictionary

    if word ends with "ing" and len > 5:  return word[:-3]
    if word ends with "ed"  and len > 4:  return word[:-2]
    if word ends with "ies" and len > 4:  return word[:-3] + "y"
    if word ends with "s"   and not "ss"
                            and len > 3:  return word[:-1]
    return word
```

### Algorithm C - Training pipeline (3-model bake-off)

```
ALGORITHM train_and_evaluate():
    intents  <- load(data/intents.json)
    learned  <- db.get_learned_patterns()      # rows from runtime ML loop

    patterns, tags <- []
    for each intent in intents:
        for each pattern in intent.patterns:
            patterns.append(clean_text(pattern))
            tags.append(intent.tag)
    for each row in learned:
        patterns.append(clean_text(row.pattern))
        tags.append(row.intent)

    vectorizer <- TfidfVectorizer(max_features=500)
    X <- vectorizer.fit_transform(patterns)
    y <- numpy.array(tags)

    X_train, X_test, y_train, y_test
        <- train_test_split(X, y, test_size=0.2,
                            random_state=42, stratify=y)

    candidates <- {
        "Naive Bayes":   MultinomialNB(),
        "SVM":           SVC(kernel='linear', probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    best_name, best_cv <- None, 0
    for name, model in candidates:
        model.fit(X_train, y_train)
        cv_scores <- cross_val_score(model, X, y, cv=5)
        if mean(cv_scores) > best_cv:
            best_cv <- mean(cv_scores)
            best_name <- name

    pickle.dump(candidates[best_name], "models/chatbot_model.pkl")
    pickle.dump(vectorizer, "models/vectorizer.pkl")
    db.mark_patterns_used()
    return best_name
```

### Algorithm D - Feedback-driven ML loop

```
ALGORITHM record_feedback(message, response, predicted, conf,
                          helpful, expected):
    db.log_feedback(message, response, predicted, conf,
                    helpful, expected)

    if not helpful and expected is not None:
        db.add_learned_pattern(message, expected, "feedback")
        if db.count_pending_patterns() >= AUTO_RETRAIN_THRESHOLD (5):
            train_and_evaluate()       # full retrain
            return retrained = True

    return retrained = False
```

### Flowchart (text form)

```
                +---------------+
                | User message  |
                +-------+-------+
                        |
                        v
                +---------------+
                |  clean_text   |
                +-------+-------+
                        |
                        v
                +---------------+
                | TF-IDF vector |
                +-------+-------+
                        |
                        v
                +-----------------+
                | SVM.predict +   |
                | predict_proba   |
                +-------+---------+
                        |
                  conf >= 0.4 ?
                  /         \
                yes          no
                /              \
               v                v
   +-------------------+   +------------+
   |  tag in DYNAMIC?  |   | tag <-     |
   +-------+---------+ |   | "fallback" |
           |          | |   +------+----+
          yes         no |          |
           |          |  +----------+
           v          v             |
  +----------------+  +---------+   |
  | DB query +     |  | random  |<--+
  | format answer  |  | static  |
  +-------+--------+  +----+----+
          |                |
          +-------+--------+
                  v
       +---------------------+
       | log_chat_history    |
       +---------------------+
                  v
       +---------------------+
       |  return JSON to UI  |
       +---------------------+
```

---

## 7. Key design ideas with code snippets

This section maps each row in the marking scheme's "key design ideas
supported by code snippets" line (8 marks) to its implementation.

### 7.1 Lemmatising

`app/preprocess.py` implements lemmatisation without an NLTK download:

```python
LEMMA_RULES = {
    'courses': 'course', 'programs': 'program', 'fees': 'fee',
    'exams': 'exam', 'examination': 'exam',
    'applying': 'apply', 'applied': 'apply', 'applies': 'apply',
    # ...about 50 rules
}

def simple_lemmatize(word):
    if word in LEMMA_RULES:
        return LEMMA_RULES[word]
    if word.endswith('ing') and len(word) > 5:
        return word[:-3]
    if word.endswith('ed') and len(word) > 4:
        return word[:-2]
    if word.endswith('ies') and len(word) > 4:
        return word[:-3] + 'y'
    if word.endswith('s') and not word.endswith('ss') and len(word) > 3:
        return word[:-1]
    return word
```

`clean_text` orchestrates the full pipeline (lower-case → strip
punctuation → tokenise → lemmatise → drop stopwords).

### 7.2 Small talk

The four "static" intents (`greeting`, `goodbye`, `thanks`,
`fallback`) live in `data/intents.json`. They are answered without
ever touching the database:

```python
# app/chat.py
DYNAMIC_INTENTS = {
    'courses', 'fees', 'admission', 'scholarship', 'exams',
    'timetable', 'library', 'contact', 'faculty', 'hostel', 'events'
}

def _build_response(self, tag):
    if tag in DYNAMIC_INTENTS:
        ...                     # try DB first
    if tag in self.response_map:
        return random.choice(self.response_map[tag]), 'static'
    return ("I'm sorry, ...", 'fallback')
```

### 7.3 Random answers

Static intents have multiple template responses; the bot picks one at
random so that consecutive greetings vary. From `app/chat.py`:

```python
import random
response = random.choice(self.response_map[tag])
```

`intents.json` carries 3-5 responses per static intent, e.g. four
different goodbye lines so the bot doesn't sound robotic.

### 7.4 Getting database answers

Every dynamic intent has its own builder that issues a SQL query and
formats the rows into a friendly reply. Example - `_respond_courses`:

```python
@staticmethod
def _respond_courses():
    rows = db.list_courses()              # SQL: SELECT * FROM courses
    if not rows:
        return None                        # falls back to template
    ug = [r for r in rows if r['level'] == 'Undergraduate']
    pg = [r for r in rows if r['level'] == 'Postgraduate']
    lines = ["Here are our current programmes:"]
    if ug:
        lines.append("\nUndergraduate:")
        for r in ug:
            lines.append(f"  - {r['name']} ({r['code']}) - "
                         f"{r['faculty']} - ${r['fee_per_year']}/year")
    if pg:
        lines.append("\nPostgraduate:")
        for r in pg:
            lines.append(f"  - {r['name']} ({r['code']}) - "
                         f"{r['faculty']} - ${r['fee_per_year']}/year")
    lines.append("\nWould you like details about any specific programme?")
    return "\n".join(lines)
```

The DB layer itself is `app/database.py`. It uses parameterised
queries throughout so we are safe against SQL injection:

```python
def find_course(keyword):
    pattern = f"%{keyword.lower()}%"
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM courses "
            "WHERE LOWER(name) LIKE ? OR LOWER(code) LIKE ? "
            "OR LOWER(faculty) LIKE ?",
            (pattern, pattern, pattern)
        )]
```

### 7.5 Training the Bot (initial training + runtime learning)

Initial training (`app/train.py`) is a 3-model bake-off picked by
5-fold cross-validation:

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
pickle.dump(models[best_name], open('models/chatbot_model.pkl', 'wb'))
```

#### Runtime learning with a two-tier trust model

A naive design would let any visitor map any phrase to any intent,
which makes the bot trivial to poison. EduBot v3 splits the learning
loop into two tiers:

```
TIER 1 (open to all chat users)
   thumbs-down + suggested intent  ──►  learned_patterns row with
                                        approved = 0 (PENDING REVIEW)

TIER 2 (admin-only, password-protected)
   admin clicks Approve  ──►  approved = 1  (eligible for training)
   admin clicks Discard  ──►  row deleted
   admin uses Teach form ──►  approved = 1 immediately

train.py consumes ONLY approved rows.
```

The trust split is enforced at three layers:

1. **HTTP layer** - `@admin_required` decorator on `/admin`, `/teach`,
   `/retrain`, `/admin/approve/<id>`, `/admin/discard/<id>`.
   Reads `EDUBOT_ADMIN_PASSWORD` from the environment; if set,
   demands HTTP Basic Auth.
2. **Application layer** - `learning.record_feedback` writes user
   suggestions with `approved=False`. `learning.teach`
   (admin-direct) writes with `approved=True`. They cannot be
   confused.
3. **Database layer** - `learned_patterns.approved INTEGER DEFAULT 0`
   plus `train.py` calling `db.get_learned_patterns(approved_only=True)`.

End-user feedback loop:

```python
# app/learning.py
def record_feedback(user_message, bot_response, predicted_intent,
                    confidence, helpful, expected_intent=None):
    db.log_feedback(user_message, bot_response, predicted_intent,
                    confidence, helpful, expected_intent)
    if not helpful and expected_intent:
        db.add_learned_pattern(
            pattern=user_message,
            intent=expected_intent,
            source='feedback_correction',
            approved=False,            # ← waits for admin review
        )
    return {'retrained': False,
            'pending_review': db.count_pending_review(),
            'learned_patterns_pending': db.count_pending_patterns()}
```

Admin curation:

```python
def approve_suggestion(pattern_id):
    db.approve_pattern(pattern_id)        # flip approved to 1
    retrained = _maybe_auto_retrain()      # may retrain if threshold met
    return {'approved': True, 'retrained': retrained, ...}

def discard_suggestion(pattern_id):
    db.discard_pattern(pattern_id)         # DELETE from learned_patterns
    return {'discarded': True, ...}
```

Training filter:

```python
# app/train.py
learned_rows = db.get_learned_patterns(approved_only=True)
patterns, tags = prepare_training_data(intents_data, learned_rows)
```

This satisfies the marking-scheme requirement *"an indication of
machine learning - chatbot updating its own knowledge base"* while
adding a defensible answer to the obvious viva question
*"what stops a malicious user from corrupting your bot?"*

---

## 8. UML diagrams

### 8.1 Class diagram

```
+---------------------+         +-----------------------+
|       EduBot        |1       1|   TfidfVectorizer     |
|---------------------|         |  (sklearn)            |
| - model             |<------->|  vocabulary_, idf_    |
| - vectorizer        |         +-----------------------+
| - intents_data      |
| - response_map      |         +-----------------------+
| - confidence_thr    |1       1|   Classifier          |
|---------------------|<------->|   (NB | SVM | RF)     |
| + predict_intent()  |         +-----------------------+
| + get_response()    |
| - _build_response() |         +-----------------------+
| - _db_response()    |1       *|   intent  (JSON)      |
| - _respond_*()      |---------|  tag, patterns,       |
+----------+----------+         |  responses[]          |
           |                    +-----------------------+
           | uses
           v
+---------------------+         +-----------------------+
|     database (.py)  |         |   sqlite3.Connection  |
|---------------------|<------->|                       |
| init_schema()       |         +-----------------------+
| list_courses()                                         
| list_faculty()                                         
| list_events()                                         
| list_exams()                                          
| list_scholarships()                                    
| list_hostel_rooms()                                    
| get_facts_by_cat()                                     
| log_chat()                                            
| log_feedback()                                        
| add_learned_pattern()                                 
| get_learned_patterns()                                
| mark_patterns_used()                                   
| count_pending_patterns()                               
| stats()                                                
+---------------------+

           used by

+---------------------+         +-----------------------+
|     learning (.py)  |1       *|   feedback (table)    |
|---------------------|         |   user_message,       |
| AUTO_RETRAIN_THRESHOLD         |   helpful, expected,  |
|                                |   created_at          |
| record_feedback()              +-----------------------+
| teach()                                                 
| manual_retrain()               +-----------------------+
| _maybe_auto_retrain()1       *|  learned_patterns     |
| _run_training() ----------->|  pattern, intent,     |
+---------------------+        |  used_in_model        |
                               +-----------------------+
```

(Mermaid-renderable version is in `docs/diagrams/class.mmd`.)

### 8.2 State transition diagram

The bot is a finite-state machine cycling between four states. The
state is implicit (per turn) rather than carried over conversation
history, but the transitions explicitly drive the response builder
and the ML loop.

```
              +-------------+
              |    IDLE     |  <-- initial state, model loaded
              +------+------+
                     |
                     | message arrives (POST /chat)
                     v
              +-------------+
              | CLASSIFYING |
              +------+------+
                     |
                     | predict_intent() returns (tag, conf)
                     v
              +-----------------+
              |  ROUTING        |
              +--------+--------+
                       |
            conf<0.4   |   conf>=0.4 & DYNAMIC  
            -----------+-----------+--------+
            |          |           |        |
            v          |           v        v
     +-------------+   |     +----------+ +-------+
     |  FALLBACK   |   |     | DB QUERY | | STATIC|
     +------+------+   |     +-----+----+ +---+---+
            |          |           |          |
            +----------+-----------+----------+
                       |
                       v
              +---------------+
              |  RESPONDING   |  log + return JSON
              +------+--------+
                     |
                     | thumbs-down + expected (optional)
                     v
              +-----------------+
              |  LEARNING       |
              +--------+--------+
                       |
              pending >= 5 ?
                  /        \
                yes         no
                /            \
               v              v
         +-----------+   +-------+
         | RETRAINING |   | IDLE  |
         +-----+-----+    +-------+
               |
               v
            +-------+
            | IDLE  |  (with new model swapped in)
            +-------+
```

### 8.3 Sequence diagram - one chat turn with feedback

```
User      Browser       Flask         EduBot       database     learning
 |          |             |              |             |            |
 | type     |             |              |             |            |
 |--------->|             |              |             |            |
 |          | POST /chat  |              |             |            |
 |          |------------>|              |             |            |
 |          |             | get_response |             |            |
 |          |             |------------->|             |            |
 |          |             |              |  list_courses           |
 |          |             |              |------------>|            |
 |          |             |              |<------------|            |
 |          |             |              | log_chat    |            |
 |          |             |              |------------>|            |
 |          |             |<-------------|             |            |
 |          |<------------|              |             |            |
 |<---------|             |              |             |            |
 | render   |             |              |             |            |
 |          |             |              |             |            |
 | thumbs-down + expected |             |             |            |
 |--------->|             |              |             |            |
 |          | POST /feedback             |             |            |
 |          |------------>| record_feedback             |           |
 |          |             |---------------------------->|           |
 |          |             |             | log_feedback |            |
 |          |             |             |------------->|            |
 |          |             |             | add_learned_pattern        |
 |          |             |             |------------->|            |
 |          |             |             | (>=5 pending? -> retrain)  |
 |          |<------------|             |              |            |
 |<---------|             |              |             |            |
```

---

## 9. Source code listing

The full source tree (excluding `venv/` and `__pycache__/`) is:

```
edubot-v3/
├── README.md
├── app.py                              Flask server + REST routes
├── build.spec                          PyInstaller config
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── preprocess.py                   tokenise + lemmatise + stopwords
│   ├── database.py                     SQLite schema + read/write helpers
│   ├── seed_db.py                      initial knowledge-base data
│   ├── train.py                        3-model training pipeline
│   ├── chat.py                         inference engine + DB-backed responses
│   └── learning.py                     feedback-driven retraining loop
├── data/
│   ├── intents.json                    static patterns + small-talk responses
│   └── edubot.db                       SQLite (auto-generated)
├── models/
│   ├── chatbot_model.pkl               pickled SVM
│   ├── vectorizer.pkl                  pickled TF-IDF vectoriser
│   └── model_info.txt                  human-readable metadata
├── templates/
│   ├── index.html                      chat UI
│   └── admin.html                      admin dashboard
├── static/
│   ├── script.js                       chat UI + feedback (👍/👎)
│   └── style.css                       all styling
├── tests/
│   └── test_smoke.py                   pytest smoke tests
└── docs/
    └── TECHNICAL_DOCUMENTATION.md      this document
```

Each `.py` file carries a docstring header explaining its purpose, and
inline comments highlight non-obvious logic (e.g. why the SVM uses
`probability=True`, why the lemmatiser is hand-built rather than
NLTK-based). The full annotated listing is the source code itself -
all files in `app/`, `tests/`, plus `app.py` and `build.spec`.

---

## 10. Test plan, test data and results

### 10.1 Test plan

| ID | Tier | What is being tested | Method |
|----|------|----------------------|--------|
| T1 | Tier 1 (NLP) | `clean_text` strips punctuation, lower-cases, lemmatises plurals | pytest unit test |
| T2 | Tier 1 (NLP) | `clean_text` keeps interrogative words despite stopword list | pytest unit test |
| T3 | Tier 3 (DB)  | `seed_db.seed_all` populates all required tables with the expected counts | pytest integration test |
| T4 | Tier 3 (DB)  | DB read helpers (`list_courses`, `get_fact`) return dict rows with expected fields | pytest integration test |
| T5 | Tier 2 (Inference) | "What courses do you offer?" classifies as `courses` with conf ≥ 0.4 | pytest integration test |
| T6 | Three-tier  | Course query returns answer with `source = 'database'` and includes a real course name | pytest end-to-end |
| T7 | Three-tier  | Greeting returns `source = 'static'` (does not hit DB) | pytest end-to-end |
| T8 | Inference   | Gibberish ("zzz qwerty asdfgh") falls back gracefully | pytest end-to-end |
| T9 | ML loop     | `learning.teach()` adds a row, `manual_retrain()` consumes it (count_pending == 0) | pytest end-to-end |
| T10 | API        | `/health` returns 200 OK | manual curl |
| T11 | API        | `/chat` returns valid JSON for courses, fees, greeting | manual curl |
| T12 | API        | `/feedback` (thumbs-down + expected) increments pending count | manual curl |
| T13 | API        | `/teach` accepts (pattern, intent), rejects unknown intent | manual curl |
| T14 | API        | `/retrain` returns the new winning model name | manual curl |

### 10.2 Test data (representative phrases per intent)

| Intent | Sample test inputs |
|--------|-----------------|
| greeting    | "Hi", "Good morning", "Hellooo", "Hey can you help" |
| goodbye     | "Bye", "Talk to you later", "Thanks bye" |
| thanks      | "Thank you", "Tysm", "That's helpful" |
| courses     | "What courses do you offer", "Tell me about programs", "Computer science course" |
| admission   | "How do I apply", "When is the deadline", "Documents required" |
| fees        | "What are the fees", "How much is BSc CS", "Per semester cost" |
| scholarship | "Do you offer scholarships", "Merit scholarship", "Help with fees" |
| exams       | "When are exams", "Final exam dates", "Grading system" |
| timetable   | "Class schedule", "When is my next lecture" |
| library     | "Library hours", "Where is the library", "Borrow books" |
| contact     | "Phone number", "Email address", "How to reach you" |
| faculty     | "Who is the dean", "Lecturers list", "Office hours" |
| hostel      | "Is hostel available", "Single room price", "Hostel facilities" |
| events      | "Upcoming events", "Hackathon", "Career fair" |
| fallback    | "asdfgh", "lorem ipsum", "qwerty xyz" |

### 10.3 Automated test results

`python -m pytest tests/ -v` (run on 2026-05-03):

```
tests/test_smoke.py::test_clean_text_strips_punctuation_and_lemmatises  PASSED
tests/test_smoke.py::test_clean_text_keeps_question_words               PASSED
tests/test_smoke.py::test_db_seeding_populates_required_tables          PASSED
tests/test_smoke.py::test_db_helpers_return_dicts                       PASSED
tests/test_smoke.py::test_predict_intent_routes_courses_to_courses_tag  PASSED
tests/test_smoke.py::test_get_response_uses_database_for_courses        PASSED
tests/test_smoke.py::test_get_response_uses_static_for_greeting         PASSED
tests/test_smoke.py::test_low_confidence_falls_back                     PASSED
tests/test_smoke.py::test_teach_adds_learned_pattern_and_can_retrain    PASSED

============================== 9 passed in 4.29s ==============================
```

### 10.4 Training output

Output from `python app/train.py` after seeding:

```
Step 1: Loading training data...
   - 15 intents loaded from intents.json
   - 999 seed patterns
   - 0 patterns learned at runtime

Step 3: Vectorising with TF-IDF...
   - Feature matrix shape: (999, 500)
   - Vocabulary size: 500

Step 5: Training models...
   Naive Bayes  : Train 91.4%  Test 82.5%  CV 77.3% (+/-3.6%)
   SVM          : Train 96.7%  Test 84.0%  CV 78.4% (+/-3.8%)
   Random Forest: Train 98.5%  Test 77.0%  CV 75.7% (+/-5.2%)

   Best Model: SVM (CV accuracy: 78.4%)
```

Per-intent F1 (Linear SVM, held-out test set):

| Intent      | Precision | Recall | F1   | Support |
|-------------|-----------|--------|------|--------:|
| admission   | 1.00      | 0.93   | 0.97 |     15  |
| contact     | 0.86      | 0.92   | 0.89 |     13  |
| courses     | 0.78      | 0.88   | 0.82 |     16  |
| events      | 0.64      | 0.64   | 0.64 |     14  |
| exams       | 0.93      | 0.87   | 0.90 |     15  |
| faculty     | 0.81      | 1.00   | 0.90 |     13  |
| fees        | 1.00      | 0.84   | 0.91 |     19  |
| goodbye     | 0.90      | 0.82   | 0.86 |     11  |
| greeting    | 0.44      | 0.79   | 0.56 |     14  |
| hostel      | 1.00      | 0.79   | 0.88 |     14  |
| library     | 1.00      | 0.93   | 0.96 |     14  |
| scholarship | 0.93      | 0.93   | 0.93 |     14  |
| thanks      | 1.00      | 0.80   | 0.89 |     10  |
| timetable   | 0.92      | 0.92   | 0.92 |     13  |
| **Overall** | **0.85**  | **0.84** | **0.84** | **200** |

The `events` and `greeting` rows have weaker F1 because their
patterns overlap with the broader "general question" phrasing -
discussed as a known limitation in section 11.

### 10.5 Manual API regression results

```
GET  /health                 -> 200 {"status":"ok"}
POST /chat ("courses?")      -> 200, conf=0.996, source=database, response includes "BSc Computer Science"
POST /chat ("how much?")     -> 200, conf=0.962, source=database, includes "$3000/year"
POST /chat ("hello")         -> 200, conf=0.785, source=static
POST /chat ("zzz qwerty")    -> 200, tag=fallback
POST /feedback (thumbs up)   -> 200, retrained=false, pending=0
POST /feedback (down + fees) -> 200, retrained=false, pending=1
POST /teach (fees pattern)   -> 200, retrained=false, pending=2
POST /retrain                -> 200, retrained=true, model=SVM
```

---

## 11. Conclusion

EduBot v3 is a complete three-tier conversational assistant that meets
every line on the assignment marking scheme for Option 2 (Chat Bot).

What has been delivered:

- A trained, production-grade NLP intent classifier with 84% test
  accuracy and 78% cross-validated accuracy across 14 intents
  (Linear SVM picked from a 3-model bake-off).
- A SQLite knowledge base whose contents - course list, fees,
  faculty, events, exams, scholarships, hostel options - are queried
  live to answer dynamic questions.
- A two-tier feedback-driven machine-learning loop: end users flag
  wrong answers and suggest intents (Tier 1, open); admins curate
  and approve those suggestions before they enter training (Tier 2,
  admin-only). The trust split is enforced at the HTTP layer
  (Basic Auth via `EDUBOT_ADMIN_PASSWORD`), the application layer
  (`learning.record_feedback` writes `approved=False`,
  `learning.teach` writes `approved=True`), and the database layer
  (`train.py` only loads `approved=1` rows).
- A Flask web service exposing a chat UI, an admin dashboard, and a
  set of REST endpoints (`/chat`, `/feedback` open to all users;
  `/teach`, `/retrain`, `/admin`, `/admin/approve/<id>`,
  `/admin/discard/<id>` admin-only; `/api/stats`, `/api/intents`,
  `/health` for diagnostics).
- A PyInstaller spec that bundles the entire application into a
  single Windows executable, satisfying the brief's
  *"runs from an executable file without extra installation of
  libraries"* requirement.
- An automated test suite (9 pytest tests, all passing) covering all
  three tiers and the ML loop.

### Known limitations and future work

- The classifier does not handle compound questions (*"what are the
  fees and when do I apply?"*) - it picks the single most likely
  intent. A natural extension is multi-label classification or a
  rule-based question splitter.
- Retraining blocks the Flask request thread for a few seconds.
  Production deployment should move it to a background worker.
- The lemmatiser is rule-based; a true WordNet-backed lemmatiser
  would handle irregular forms more reliably at the cost of bundle
  size.
- The admin dashboard has no authentication. For a multi-user
  deployment, basic-auth or session login would be required.
- Confidence scores from `SVC` come from Platt scaling and can be
  miscalibrated; switching to a `CalibratedClassifierCV` wrapper
  would improve the 0.4 fallback threshold's reliability.
- The chat is single-turn (no conversation memory). A slot-filling
  layer would let the bot ask follow-up questions, e.g.
  *"Which course do you mean - BSc CS or BSc IT?"*.

### Lessons learned

1. **Splitting the responses into static/dynamic was the single
   biggest design decision.** It made the code in `chat.py`
   straightforward and turned the marking-scheme requirement for a
   database into a clean architectural feature instead of a checkbox.
2. **A small, hand-built lemmatiser was good enough.** We avoided a
   30 MB NLTK download and the bundled executable stays small.
3. **Cross-validation is essential.** Random Forest had the highest
   training accuracy but the worst CV - exactly the over-fitting
   trap CV is designed to catch.

---

## 12. References

1. Manning, C. D., Raghavan, P. & Schuetze, H. (2008).
   *Introduction to Information Retrieval*. Cambridge University Press.
   Chapter 6 (scoring, term weighting and the vector-space model)
   informs the TF-IDF design.

2. Joachims, T. (1998). *Text Categorization with Support Vector
   Machines: Learning with Many Relevant Features*. Proc. ECML 1998,
   pp. 137-142. Justifies the choice of Linear SVM for sparse
   high-dimensional text features.

3. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1),
   pp. 5-32. Background for the third candidate model.

4. Kohavi, R. (1995). *A Study of Cross-Validation and Bootstrap for
   Accuracy Estimation and Model Selection*. Proc. IJCAI 1995,
   pp. 1137-1145. Justifies the 5-fold stratified CV strategy used
   to pick the winning model.

5. Jurafsky, D. & Martin, J. H. (2024). *Speech and Language
   Processing* (3rd ed. draft). Chapter 24 (chatbots and dialogue
   systems) covers the three-tier UI / NLU / KB architecture used
   here.

6. Russell, S. & Norvig, P. (2020). *Artificial Intelligence: A
   Modern Approach* (4th ed.). Pearson. Chapter 2 introduces the
   P.E.A.S. specification used in section 4.

7. scikit-learn developers. *scikit-learn user guide*. Available at
   https://scikit-learn.org/stable/user_guide.html (accessed
   2026-04). Specifically: `feature_extraction.text.TfidfVectorizer`,
   `naive_bayes.MultinomialNB`, `svm.SVC`,
   `ensemble.RandomForestClassifier`,
   `model_selection.cross_val_score`.

8. Hipp, D. R. *The Architecture of SQLite*. SQLite Foundation,
   https://sqlite.org/arch.html (accessed 2026-04). Background for
   the choice of file-based DB.

9. Grinberg, M. (2018). *Flask Web Development* (2nd ed.). O'Reilly.
   Reference for the routing and CORS patterns used in `app.py`.

10. Pallets Projects. *Flask documentation*.
    https://flask.palletsprojects.com/ (accessed 2026-04).

11. PyInstaller team. *PyInstaller manual*.
    https://pyinstaller.org/en/stable/ (accessed 2026-04). Used for
    the standalone executable build.

12. London Metropolitan University. *Artificial Intelligence -
    Assessment 2026 brief.* (Module assignment document.)
