# EduBot - Diagrams (Mermaid source)

This file contains **Mermaid** source for every diagram listed in
the assignment brief, in the priority order from the spec sheet.

## How to render

1. Open https://mermaid.live/ in your browser.
2. Copy the code inside any ```` ```mermaid ```` block below.
3. Paste it into the left-hand editor pane.
4. The right pane renders the diagram instantly. Use **Actions ->
   PNG / SVG** to export.

For the report, export each diagram as PNG (300 dpi recommended)
and embed it in the corresponding section of
`TECHNICAL_DOCUMENTATION.md`.

---

## Priority order (from the assignment brief)

1. [System Architecture](#1-system-architecture-must-have)
2. [Activity Diagram](#3-activity-diagram-must-have)
3. [Use Case Diagram](#2-use-case-diagram-must-have)
4. [Sequence Diagram](#4-sequence-diagram-must-have)
5. [Feedback Learning Loop](#9-feedback-learning-loop-very-important)
6. [ER Diagram](#7-er-diagram-very-important)
7. [Class Diagram](#5-class-diagram-must-have)
8. [State Transition](#6-state-transition-diagram-must-have)
9. [ML Pipeline Flowchart](#8-ml-pipeline-flowchart-very-important)
10. [Deployment Diagram](#10-deployment-diagram-bonus)

---

## 1. System Architecture (MUST HAVE)

The canonical three-tier view: UI -> inference engine -> knowledge
base. Mirrors the architecture description in §3 of the technical
documentation.

```mermaid
flowchart TB
    subgraph T1["Tier 1 - Natural Language Interface"]
        direction LR
        ChatUI["index.html<br/>script.js<br/>style.css"]
        Mood["Mood avatar<br/>happy / neutral / confused / thinking"]
        Voice["Voice mic + read-aloud<br/>(Web Speech API)"]
        AdminUI["admin.html"]
        ChatUI --- Mood
        ChatUI --- Voice
    end

    subgraph T2["Tier 2 - Inference Engine"]
        direction TB
        Flask["app.py<br/>Flask routes"]
        Validate["validate.py<br/>quality gate"]
        Chat["chat.py - EduBot<br/>predict + keyword rescue<br/>per-entity responses"]
        Context["context.py<br/>sessions + anaphora"]
        Preprocess["preprocess.py<br/>clean_text"]
        Train["train.py<br/>3-model bake-off"]
        Learn["learning.py<br/>two-tier trust ML loop"]
        Flask --> Validate
        Flask --> Chat
        Chat --> Context
        Chat --> Preprocess
        Flask --> Learn
        Learn --> Train
    end

    subgraph T3["Tier 3 - Knowledge Base"]
        direction LR
        Intents[("intents.json<br/>static patterns")]
        SQLite[("edubot.db (SQLite)<br/>courses, faculty, events,<br/>exams, scholarships,<br/>hostel_rooms, kv_facts,<br/>feedback, learned_patterns,<br/>chat_history")]
        Models["models/*.pkl<br/>TF-IDF + SVM"]
    end

    User(["User browser"]) -- "HTTPS<br/>JSON" --> Flask
    Flask -- "render" --> User
    Chat -- "queries" --> SQLite
    Chat -- "templates" --> Intents
    Chat -- "loads" --> Models
    Train -- "writes" --> Models
    Train -- "reads" --> Intents
    Train -- "reads approved" --> SQLite
    Learn -- "writes" --> SQLite
```

---

## 2. Use Case Diagram (MUST HAVE)

Two actors (Student and Admin) and the use cases they trigger.

Mermaid does not have a native UML use-case shape, so we use a
left-to-right flowchart with rounded actors and oval use cases -
this is the conventional Mermaid workaround and renders cleanly.

```mermaid
flowchart LR
    Student(("Student"))
    Admin(("Admin"))

    subgraph EduBot["EduBot system"]
        UC1(["Ask university question"])
        UC2(["Receive AI answer"])
        UC3(["Use voice input (mic)"])
        UC4(["Hear reply read aloud"])
        UC5(["Click sidebar topic shortcut"])
        UC6(["Give thumbs-up feedback"])
        UC7(["Give thumbs-down<br/>+ suggest correct intent"])
        UC8(["Reset session memory"])

        UC9(["Review pending suggestions"])
        UC10(["Approve suggestion"])
        UC11(["Discard suggestion"])
        UC12(["Teach bot directly"])
        UC13(["Force model retrain"])
        UC14(["View stats / feedback log"])
    end

    Student --> UC1
    Student --> UC2
    Student --> UC3
    Student --> UC4
    Student --> UC5
    Student --> UC6
    Student --> UC7
    Student --> UC8

    Admin --> UC9
    Admin --> UC10
    Admin --> UC11
    Admin --> UC12
    Admin --> UC13
    Admin --> UC14

    UC7 -. "produces pending pattern" .-> UC9
    UC10 -. "may trigger" .-> UC13
```

---

## 3. Activity Diagram (MUST HAVE)

End-to-end activity for a single chat turn (`POST /chat`). Covers
quality validation, anaphora resolution, the clarify branch, the
keyword rescue, entity extraction, and the four response paths.

```mermaid
flowchart TD
    Start(["User submits message"]) --> Validate{"Quality<br/>validation passes?"}
    Validate -- "no (gibberish / phone / symbols)" --> Err["Return HTTP 400<br/>show inline error"]
    Err --> End1(["End"])

    Validate -- "yes" --> Resolve["resolve_pronouns<br/>using session memory"]
    Resolve --> Clarify{"Pronoun used<br/>but no last_entity?"}
    Clarify -- "yes" --> Ask["Reply: Which programme<br/>would you like to know about?"]
    Ask --> End2(["End"])

    Clarify -- "no" --> Clean["clean_text:<br/>lower, strip, lemmatise,<br/>drop stopwords"]
    Clean --> Vec["TF-IDF.transform"]
    Vec --> Predict["SVM.predict<br/>+ predict_proba"]
    Predict --> Conf{"conf >= 0.4 ?"}

    Conf -- "no" --> Rescue["match_keyword_intent"]
    Rescue --> RescueHit{"keyword hit?"}
    RescueHit -- "yes" --> Bump["tag = keyword<br/>conf = max of conf and 0.6"]
    RescueHit -- "no" --> Fallback["tag = fallback"]

    Conf -- "yes" --> Extract["extract_course_entity<br/>or carry forward<br/>last_entity on pronoun"]
    Bump --> Extract
    Fallback --> Extract

    Extract --> Promote{"entity AND<br/>tag = fallback ?"}
    Promote -- "yes" --> ToCourses["tag = courses"]
    Promote -- "no" --> Build{"Response builder?"}
    ToCourses --> Build

    Build -- "entity + fees" --> PerCourseFee["per-course fee block"]
    Build -- "entity + courses" --> PerCourseDetail["per-course detail card"]
    Build -- "dynamic intent" --> DBQuery["SQL query +<br/>format answer"]
    Build -- "static intent" --> Template["random.choice<br/>from intents.json"]
    Build -- "fallback" --> FallbackMsg["fallback template"]

    PerCourseFee --> Update["update_session +<br/>log_chat_history"]
    PerCourseDetail --> Update
    DBQuery --> Update
    Template --> Update
    FallbackMsg --> Update

    Update --> Resp["Return JSON:<br/>tag, response, conf,<br/>source, entity, session_id"]
    Resp --> End3(["End"])
```

---

## 4. Sequence Diagram (MUST HAVE)

One `POST /chat` turn from user to rendered reply, showing every
collaborator. Use this in §8.3 of the technical documentation.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Browser
    participant Flask as Flask (app.py)
    participant V as validate.py
    participant Ctx as context.py
    participant Bot as chat.py - EduBot
    participant DB as database.py / SQLite

    User->>Browser: type message + click send
    Browser->>Flask: POST /chat {message, session_id}

    Flask->>V: clean_text_field + check_message_quality
    alt invalid
        V-->>Flask: ValidationError
        Flask-->>Browser: HTTP 400 {error}
        Browser-->>User: render error inline
    else valid
        V-->>Flask: cleaned text
    end

    Flask->>Ctx: get_session(session_id)
    Ctx-->>Flask: session

    Flask->>Bot: get_response(text, session)
    Bot->>Ctx: has_pronoun + resolve_pronouns
    Ctx-->>Bot: resolved_text

    alt pronoun without last_entity
        Bot-->>Flask: clarify response
    else
        Bot->>Bot: clean_text + TF-IDF.transform
        Bot->>Bot: SVM.predict + predict_proba

        alt conf below 0.4
            Bot->>Bot: _match_keyword_intent
            Note over Bot: tag = keyword, conf = 0.6
        end

        Bot->>Bot: _extract_course_entity
        Note over Bot: promote fallback to courses<br/>if entity is set

        alt entity + fees
            Bot->>DB: list_courses (filter by name)
        else dynamic intent
            Bot->>DB: list_courses / list_events / ...
        else static intent
            Bot->>Bot: random.choice(intents.json)
        end
        DB-->>Bot: rows

        Bot->>Ctx: update_session(entity, intent)
        Bot->>DB: log_chat
    end

    Bot-->>Flask: {tag, response, conf, source, entity}
    Flask-->>Browser: JSON response
    Browser-->>User: render bubble + mood avatar +<br/>(optional) speak via speechSynthesis
```

---

## 5. Class Diagram (MUST HAVE)

Logical view of the inference engine and its collaborators. Module-
level helpers are stereotyped `<<module>>`; database tables are
shown as anchored boxes.

```mermaid
classDiagram
    class EduBot {
        -model
        -vectorizer
        -intents_data
        -response_map
        -confidence_threshold
        +predict_intent(text) tuple
        +get_response(text, session) dict
        -_build_response(tag, entity) tuple
        -_extract_course_entity(text) name
        -_db_response(tag) str
        -_respond_courses()
        -_respond_fees()
        -_respond_admission()
        -_respond_scholarship()
        -_respond_exams()
        -_respond_timetable()
        -_respond_library()
        -_respond_contact()
        -_respond_faculty()
        -_respond_hostel()
        -_respond_events()
        -_respond_course_detail(name)
        -_respond_fees_for_course(name)
    }

    class Context {
        <<module>>
        -_SESSIONS dict
        -SESSION_TTL_SECONDS int
        -MAX_SESSIONS int
        +new_session_id() uuid
        +get_session(sid, create) dict
        +has_pronoun(text) bool
        +is_followup(text) bool
        +resolve_pronouns(text, session) str
        +update_session(session, kwargs)
        +reset_session(sid)
    }

    class Validate {
        <<module>>
        +clean_text_field(value, field, check_quality) str
        +check_message_quality(text, field)
        +parse_bool(value, field) bool
        +parse_float_in_range(value, field, lo, hi) float
        +validate_intent(value, field, allowed) str
        +ValidationError
    }

    class Preprocess {
        <<module>>
        +clean_text(text) str
        +simple_lemmatize(word) str
        +preprocess_patterns(patterns) list
    }

    class Database {
        <<module>>
        +init_schema()
        +list_courses() list
        +find_course(keyword) list
        +list_events() list
        +list_faculty() list
        +get_dean() dict
        +list_exams() list
        +list_scholarships() list
        +list_hostel_rooms() list
        +get_fact(key) str
        +get_facts_by_category(cat) dict
        +log_chat(message, response, intent, conf, source)
        +log_feedback(message, response, intent, conf, helpful, expected)
        +add_learned_pattern(pattern, intent, source, approved)
        +approve_pattern(id) bool
        +discard_pattern(id) bool
        +get_learned_patterns(approved_only, only_unused) list
        +get_pending_patterns() list
        +mark_patterns_used()
        +count_pending_patterns() int
        +count_pending_review() int
        +stats() dict
    }

    class Learning {
        <<module>>
        +AUTO_RETRAIN_THRESHOLD int
        +record_feedback(message, response, intent, conf, helpful, expected)
        +teach(pattern, intent)
        +approve_suggestion(id)
        +discard_suggestion(id)
        +manual_retrain()
        -_maybe_auto_retrain()
        -_run_training()
    }

    class Train {
        <<module>>
        +train_and_evaluate(verbose)
    }

    class KeywordRescue {
        <<chat.py helpers>>
        +_INTENT_KEYWORDS tuple
        +_match_keyword_intent(text) tag
    }

    class TfidfVectorizer {
        <<sklearn>>
    }

    class Classifier {
        <<sklearn>>
        NB or SVM or RF
    }

    EduBot --> TfidfVectorizer
    EduBot --> Classifier
    EduBot --> Context : uses
    EduBot --> Preprocess : uses
    EduBot --> Database : queries
    EduBot ..> KeywordRescue : module-level helper
    Validate <.. EduBot : called by Flask before EduBot
    Learning --> Database : reads/writes
    Learning --> Train : invokes retrain
    Train --> Database : reads approved patterns
    Train --> TfidfVectorizer : fits
    Train --> Classifier : fits
```

---

## 6. State Transition Diagram (MUST HAVE)

Per-request finite-state machine for `POST /chat`. The CLARIFY,
KEYWORD-RESCUE and per-entity branches are explicit so the diagram
matches the actual code path.

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Validating: POST /chat
    Validating --> Idle: invalid (HTTP 400)
    Validating --> Resolving: valid

    Resolving --> Clarify: pronoun + no last_entity
    Clarify --> Idle: return CLARIFY message

    Resolving --> Classifying: ok
    Classifying --> KeywordRescue: conf below 0.4
    Classifying --> EntityExtract: conf at or above 0.4

    KeywordRescue --> EntityExtract: keyword hit, conf = 0.6
    KeywordRescue --> EntityExtract: no hit, tag = fallback

    EntityExtract --> Routing
    Routing --> PerEntity: entity + dynamic intent
    Routing --> DBQuery: dynamic intent
    Routing --> Static: static intent
    Routing --> Fallback: fallback tag

    PerEntity --> Responding
    DBQuery --> Responding
    Static --> Responding
    Fallback --> Responding

    Responding --> Idle: log_chat + update_session +<br/>return JSON
```

---

## 7. ER Diagram (VERY IMPORTANT)

SQLite schema. The seed tables (courses, faculty, events, etc.)
have no foreign keys — they are joined at the application layer by
intent name. The `feedback -> learned_patterns` link is logical
(feedback records that include `expected_intent` produce a row in
`learned_patterns`).

```mermaid
erDiagram
    courses {
        INTEGER id PK
        TEXT code UK
        TEXT name
        TEXT level
        TEXT faculty
        REAL duration_years
        INTEGER fee_per_year
        TEXT description
        TEXT updated_at
    }

    faculty {
        INTEGER id PK
        TEXT title
        TEXT name
        TEXT department
        TEXT expertise
        TEXT email
        TEXT office_hours
        INTEGER is_dean
    }

    events {
        INTEGER id PK
        TEXT name
        TEXT start_date
        TEXT end_date
        TEXT location
        TEXT category
        TEXT description
    }

    exams {
        INTEGER id PK
        TEXT exam_type
        TEXT start_date
        TEXT end_date
        TEXT format
        TEXT notes
    }

    scholarships {
        INTEGER id PK
        TEXT name UK
        INTEGER max_percentage
        TEXT eligibility
        TEXT description
    }

    hostel_rooms {
        INTEGER id PK
        TEXT room_type UK
        INTEGER capacity
        INTEGER price_per_semester
        TEXT amenities
    }

    kv_facts {
        TEXT key PK
        TEXT value
        TEXT category
    }

    feedback {
        INTEGER id PK
        TEXT user_message
        TEXT bot_response
        TEXT predicted_intent
        REAL confidence
        INTEGER helpful
        TEXT expected_intent
        TEXT created_at
    }

    learned_patterns {
        INTEGER id PK
        TEXT pattern
        TEXT intent
        TEXT source
        INTEGER approved
        TEXT created_at
        INTEGER used_in_model
    }

    chat_history {
        INTEGER id PK
        TEXT user_message
        TEXT bot_response
        TEXT intent
        REAL confidence
        TEXT response_source
        TEXT created_at
    }

    feedback ||..o{ learned_patterns : "produces (when helpful=0 + expected_intent)"
    chat_history }o..|| feedback : "may receive thumbs feedback"
```

### 7-Chen. ER Diagram in Chen notation (alternative view)

The diagram above uses **Crow's Foot** notation (industry standard,
what Mermaid `erDiagram` produces natively). This second version
uses **Chen notation** (the textbook style: rectangles =
entities, ovals = attributes, diamonds = relationships).

Mermaid does not support Chen natively, so we simulate it with a
`flowchart` and shape classes. For a polished version, redraw this
in draw.io's *ER (Chen)* shape group using the spec below.

The Chen ER is split into two diagrams so each renders cleanly:

- **§7-Chen-A** - Knowledge base (7 seed entities, no relationships)
- **§7-Chen-B** - Runtime / ML (3 entities + 2 relationship diamonds)

Together they cover all 10 tables in `database.py`.

#### 7-Chen-A. Knowledge-base entities (7 read-only seed tables)

These tables have **no foreign keys to anything** - they are
queried by the bot when an intent matches their domain (e.g.
`courses` table is read when the predicted intent is `courses`).
Drawing them as isolated entity rectangles with attribute ovals is
the correct Chen rendering for read-only seed data.

```mermaid
flowchart TB
    classDef entity fill:#bfdbfe,stroke:#1e3a8a,color:#0f172a,stroke-width:2px,font-weight:bold
    classDef attr   fill:#bbf7d0,stroke:#166534,color:#0f172a
    classDef pk     fill:#fde68a,stroke:#92400e,color:#0f172a,stroke-width:2px,font-weight:bold

    %% ---------------- courses ----------------
    c_pk(("id (PK)")):::pk
    c_code(("code (UK)")):::pk
    c_name(("name")):::attr
    c_level(("level")):::attr
    c_fac(("faculty")):::attr
    c_dur(("duration_years")):::attr
    c_fee(("fee_per_year")):::attr
    c_desc(("description")):::attr
    Courses["courses"]:::entity
    c_pk --- Courses
    c_code --- Courses
    c_name --- Courses
    c_level --- Courses
    c_fac --- Courses
    c_dur --- Courses
    c_fee --- Courses
    c_desc --- Courses

    %% ---------------- faculty ----------------
    fa_pk(("id (PK)")):::pk
    fa_title(("title")):::attr
    fa_name(("name")):::attr
    fa_dept(("department")):::attr
    fa_exp(("expertise")):::attr
    fa_email(("email")):::attr
    fa_oh(("office_hours")):::attr
    fa_dean(("is_dean")):::attr
    Faculty["faculty"]:::entity
    fa_pk --- Faculty
    fa_title --- Faculty
    fa_name --- Faculty
    fa_dept --- Faculty
    fa_exp --- Faculty
    fa_email --- Faculty
    fa_oh --- Faculty
    fa_dean --- Faculty

    %% ---------------- events ----------------
    e_pk(("id (PK)")):::pk
    e_name(("name")):::attr
    e_start(("start_date")):::attr
    e_end(("end_date")):::attr
    e_loc(("location")):::attr
    e_cat(("category")):::attr
    e_desc(("description")):::attr
    Events["events"]:::entity
    e_pk --- Events
    e_name --- Events
    e_start --- Events
    e_end --- Events
    e_loc --- Events
    e_cat --- Events
    e_desc --- Events

    %% ---------------- exams ----------------
    x_pk(("id (PK)")):::pk
    x_type(("exam_type")):::attr
    x_start(("start_date")):::attr
    x_end(("end_date")):::attr
    x_fmt(("format")):::attr
    x_notes(("notes")):::attr
    Exams["exams"]:::entity
    x_pk --- Exams
    x_type --- Exams
    x_start --- Exams
    x_end --- Exams
    x_fmt --- Exams
    x_notes --- Exams

    %% ---------------- scholarships ----------------
    s_pk(("id (PK)")):::pk
    s_name(("name (UK)")):::pk
    s_pct(("max_percentage")):::attr
    s_elig(("eligibility")):::attr
    s_desc(("description")):::attr
    Scholar["scholarships"]:::entity
    s_pk --- Scholar
    s_name --- Scholar
    s_pct --- Scholar
    s_elig --- Scholar
    s_desc --- Scholar

    %% ---------------- hostel_rooms ----------------
    h_pk(("id (PK)")):::pk
    h_room(("room_type (UK)")):::pk
    h_cap(("capacity")):::attr
    h_price(("price_per_semester")):::attr
    h_amen(("amenities")):::attr
    Hostel["hostel_rooms"]:::entity
    h_pk --- Hostel
    h_room --- Hostel
    h_cap --- Hostel
    h_price --- Hostel
    h_amen --- Hostel

    %% ---------------- kv_facts ----------------
    k_pk(("key (PK)")):::pk
    k_val(("value")):::attr
    k_cat(("category")):::attr
    KV["kv_facts"]:::entity
    k_pk --- KV
    k_val --- KV
    k_cat --- KV
```

#### 7-Chen-B. Runtime / ML entities (with relationships)

These three tables are written at runtime and carry the only two
relationships in the schema:

```mermaid
flowchart TB
    classDef entity fill:#bfdbfe,stroke:#1e3a8a,color:#0f172a,stroke-width:2px,font-weight:bold
    classDef attr   fill:#bbf7d0,stroke:#166534,color:#0f172a
    classDef pk     fill:#fde68a,stroke:#92400e,color:#0f172a,stroke-width:2px,font-weight:bold
    classDef rel    fill:#1f2937,stroke:#4b5563,color:#fff,stroke-width:2px

    %% ---------------- chat_history ----------------
    ch_pk(("id (PK)")):::pk
    ch_msg(("user_message")):::attr
    ch_resp(("bot_response")):::attr
    ch_int(("intent")):::attr
    ch_conf(("confidence")):::attr
    ch_src(("response_source")):::attr
    ch_at(("created_at")):::attr
    Chat["chat_history"]:::entity
    ch_pk --- Chat
    ch_msg --- Chat
    ch_resp --- Chat
    ch_int --- Chat
    ch_conf --- Chat
    ch_src --- Chat
    ch_at --- Chat

    %% ---------------- feedback ----------------
    f_pk(("id (PK)")):::pk
    f_msg(("user_message")):::attr
    f_resp(("bot_response")):::attr
    f_pred(("predicted_intent")):::attr
    f_conf(("confidence")):::attr
    f_help(("helpful")):::attr
    f_exp(("expected_intent")):::attr
    f_at(("created_at")):::attr
    Feedback["feedback"]:::entity
    f_pk --- Feedback
    f_msg --- Feedback
    f_resp --- Feedback
    f_pred --- Feedback
    f_conf --- Feedback
    f_help --- Feedback
    f_exp --- Feedback
    f_at --- Feedback

    %% ---------------- learned_patterns ----------------
    l_pk(("id (PK)")):::pk
    l_pat(("pattern")):::attr
    l_int(("intent")):::attr
    l_src(("source")):::attr
    l_app(("approved")):::attr
    l_used(("used_in_model")):::attr
    l_at(("created_at")):::attr
    Learned["learned_patterns"]:::entity
    l_pk --- Learned
    l_pat --- Learned
    l_int --- Learned
    l_src --- Learned
    l_app --- Learned
    l_used --- Learned
    l_at --- Learned

    %% ---------------- Relationships (diamonds) ----------------
    Chat --- MayReceive{"may receive<br/>thumbs feedback<br/>1 .. 0..1"}:::rel
    MayReceive --- Feedback

    Feedback --- Produces{"produces<br/>when helpful=0<br/>+ expected_intent<br/>1 .. 0..many"}:::rel
    Produces --- Learned
```

### 7-Chen-Spec. Data dictionary for manual draw.io / ERDPlus

If you redraw the full Chen ER (all 10 entities) in **draw.io** or
**ERDPlus**, use this spec as a checklist. Each entity is a
rectangle; each attribute is an oval; key attributes have
underlined labels; relationships are diamonds with cardinality
labels.

**Entities (10) and their attributes:**

| Entity | Key attributes (underline) | Other attributes |
|---|---|---|
| `courses` | id (PK), code (UK) | name, level, faculty, duration_years, fee_per_year, description, updated_at |
| `faculty` | id (PK) | title, name, department, expertise, email, office_hours, is_dean |
| `events` | id (PK) | name, start_date, end_date, location, category, description |
| `exams` | id (PK) | exam_type, start_date, end_date, format, notes |
| `scholarships` | id (PK), name (UK) | max_percentage, eligibility, description |
| `hostel_rooms` | id (PK), room_type (UK) | capacity, price_per_semester, amenities |
| `kv_facts` | key (PK) | value, category |
| `feedback` | id (PK) | user_message, bot_response, predicted_intent, confidence, helpful, expected_intent, created_at |
| `learned_patterns` | id (PK) | pattern, intent, source, approved, created_at, used_in_model |
| `chat_history` | id (PK) | user_message, bot_response, intent, confidence, response_source, created_at |

**Relationships (2):**

| Diamond | Between | Cardinality | Trigger condition |
|---|---|---|---|
| **may receive** | `chat_history` --- `feedback` | one chat row -> 0 or 1 feedback row | user clicks 👍 or 👎 |
| **produces** | `feedback` --- `learned_patterns` | one feedback row -> 0 or 1 pattern row | helpful=0 AND expected_intent is not null |

**Floating entities (no relationships):**

The 7 seed tables (`courses`, `faculty`, `events`, `exams`,
`scholarships`, `hostel_rooms`, `kv_facts`) have no foreign keys
to any other table. In a Chen diagram they appear as isolated
entity rectangles. Their **logical** link to the runtime tables
is via the `intent` string column on `chat_history` and `feedback`
- when `intent='courses'`, the bot has answered from the `courses`
table - but this is application-layer routing, not a database
relationship, so do not draw a diamond for it.

---

## 8. ML Pipeline Flowchart (VERY IMPORTANT)

The training pipeline that runs both at initial seed time and on
every auto-retrain (when 5+ approved patterns have accumulated).

```mermaid
flowchart TD
    Start(["Trigger:<br/>5+ approved learned patterns<br/>OR admin POST /retrain"]) --> Load["Load intents.json<br/>(15 intents, 999 seed patterns)"]
    Load --> LoadL["Load approved learned_patterns<br/>(approved=1, used_in_model=0)"]
    LoadL --> Merge["Merge into single corpus<br/>(patterns, tags)"]
    Merge --> Clean["clean_text on every pattern<br/>(lemmatise + stopwords)"]
    Clean --> Split["train_test_split<br/>20% test, stratified by intent"]
    Split --> Vec["TfidfVectorizer.fit_transform<br/>max_features = 500"]

    Vec --> NB["MultinomialNB.fit"]
    Vec --> SVM["SVC(kernel='linear',<br/>probability=True).fit"]
    Vec --> RF["RandomForestClassifier<br/>(n_estimators=100).fit"]

    NB --> CV1["5-fold cross_val_score"]
    SVM --> CV2["5-fold cross_val_score"]
    RF --> CV3["5-fold cross_val_score"]

    CV1 --> Pick{"Highest mean<br/>CV accuracy?"}
    CV2 --> Pick
    CV3 --> Pick

    Pick --> Save["pickle.dump<br/>chatbot_model.pkl<br/>vectorizer.pkl"]
    Save --> Info["Write model_info.txt<br/>(train/test/CV %, pattern counts)"]
    Info --> Mark["mark_patterns_used()<br/>used_in_model = 1"]
    Mark --> Hot["Flask hot-swaps<br/>bot = EduBot()"]
    Hot --> End([New model live])
```

---

## 9. Feedback Learning Loop (VERY IMPORTANT)

The two-tier trust model: end users can suggest, but only admins
can approve, and only approval reaches the training set.

```mermaid
flowchart LR
    User((User)) -->|chat| Bot["EduBot<br/>POST /chat"]
    Bot -->|reply + conf badge| Render["Browser shows reply<br/>with thumbs-up / down"]

    Render -->|👍 thumbs-up| Up["POST /feedback<br/>helpful=true"]
    Render -->|👎 + suggested intent| Down["POST /feedback<br/>helpful=false<br/>expected_intent=X"]

    Up --> LogFB[("feedback table")]

    Down --> LogFB
    Down --> AddPending["add_learned_pattern<br/>approved = 0"]
    AddPending --> Pending[("learned_patterns<br/>approved=0<br/>PENDING REVIEW")]

    Admin((Admin)) -->|opens /admin| Review{"Review<br/>pending row"}

    Review -->|Discard| Delete["discard_pattern<br/>row deleted"]
    Review -->|Approve| Promote["approve_pattern<br/>approved = 1"]

    Promote --> Approved[("learned_patterns<br/>approved=1, used_in_model=0")]
    Approved --> Threshold{"count of<br/>approved-but-unused<br/>at or above 5?"}
    Threshold -->|no| Idle1["Wait for next approval"]
    Threshold -->|yes| Retrain["train_and_evaluate"]

    Admin -->|teach form| Direct["teach<br/>add_learned_pattern<br/>approved = 1"]
    Direct --> Approved

    Admin -->|Force retrain| Retrain

    Retrain --> NewPkl["New chatbot_model.pkl"]
    NewPkl --> Mark["mark_patterns_used<br/>used_in_model = 1"]
    Mark --> HotSwap["Flask: bot = EduBot"]
    HotSwap --> Bot

    style Pending fill:#fff4cc
    style Approved fill:#cce5ff
```

---

## 10. Deployment Diagram (BONUS)

How the live bot is hosted on PythonAnywhere. The browser handles
voice I/O entirely client-side via the Web Speech API, so those
calls never reach the server.

```mermaid
flowchart TB
    subgraph Client["End-user device"]
        direction TB
        Browser["Web browser<br/>(Chrome / Edge / Firefox)"]
        SR["Web Speech API<br/>SpeechRecognition (STT)"]
        TTS["speechSynthesis (TTS)"]
        LS["localStorage<br/>session_id, TTS preference"]
        Browser --- SR
        Browser --- TTS
        Browser --- LS
    end

    subgraph Net["Public Internet (HTTPS)"]
        TLS[["TLS 1.3"]]
    end

    subgraph PA["PythonAnywhere host"]
        direction TB
        WSGI["WSGI front-end<br/>(NGINX + uWSGI)"]
        subgraph Proc["Flask process"]
            App["app.py<br/>EduBot singleton"]
            Sessions["In-memory _SESSIONS<br/>1 hr TTL, 1000 cap"]
            App --- Sessions
        end
        subgraph Disk["Persistent disk"]
            DB[("data/edubot.db<br/>SQLite<br/>(gitignored,<br/>auto-seeded)")]
            Models[("models/*.pkl<br/>TF-IDF + SVM")]
            Intents[("data/intents.json")]
        end
    end

    subgraph GH["GitHub"]
        Repo[(github.com/KHSenevirathne/<br/>edubot-v3)]
    end

    Browser -- "HTTPS request<br/>POST /chat, /feedback, ..." --> TLS
    TLS --> WSGI
    WSGI --> App
    App -- "JSON reply" --> WSGI
    WSGI --> TLS
    TLS --> Browser

    App -- "read/write" --> DB
    App -- "load on boot" --> Models
    App -- "load on boot" --> Intents

    Repo -. "git pull (deploy)" .-> PA

    style Client fill:#e8f4fd
    style PA fill:#fff4e6
    style GH fill:#f3e8ff
```

---

## Appendix - export tips

- **Mermaid Live Editor** also lets you save the diagram as a Mermaid
  link (sharable URL) - useful for collaborating with the team-mate.
- For PNG export at presentation quality, set the export scale to
  **2x** or **3x** in the Live Editor's Actions menu.
- If a diagram is too tall for one slide, render it as SVG and crop
  in PowerPoint / Keynote.
- All diagrams above were verified against the codebase as of commit
  `4719b14` (auto-seed-on-first-boot). If the code changes
  significantly, refresh the corresponding block here first, then
  the rendered image.
