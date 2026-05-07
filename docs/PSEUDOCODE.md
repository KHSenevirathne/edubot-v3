# EduBot - Pseudocode

Algorithms behind every major operation in the bot, written in
language-neutral pseudocode. Each algorithm matches the actual
Python implementation; the file path on the right of each heading
is where the real code lives.

---

## 1. Main chat algorithm — `get_response`  &nbsp; *(app/chat.py)*

The single end-to-end function that turns a user message into a
reply. Brings together preprocessing, classification, keyword rescue,
entity extraction and the four response paths.

```
ALGORITHM get_response(user_input, session)
INPUT  : user_input  - free-text message from the browser
         session     - per-session memory dict (or None)
OUTPUT : { tag, response, confidence, source, entity }

BEGIN
    IF user_input is empty THEN
        RETURN fallback("Please type a question")

    // Multi-turn dialogue: pronoun resolution + clarify branch
    had_pronoun     <- has_pronoun(user_input)
    resolved_input  <- resolve_pronouns(user_input, session)

    IF had_pronoun AND session.last_entity IS NULL THEN
        RETURN clarify("Which programme would you like to know about?")
    END IF

    // Tier 2 — Inference Engine
    cleaned        <- clean_text(resolved_input)
    vector         <- tfidf_vectorizer.transform(cleaned)
    tag, confidence <- model.predict(vector),
                       max(model.predict_proba(vector))

    // Keyword-rescue safety net for low-confidence predictions
    IF confidence < CONFIDENCE_THRESHOLD (= 0.4) THEN
        rescued_tag <- match_keyword_intent(resolved_input)
        IF rescued_tag IS NOT NULL THEN
            tag        <- rescued_tag
            confidence <- max(confidence, 0.6)
        ELSE
            tag        <- "fallback"
        END IF
    END IF

    // Per-entity routing (multi-turn consulting flow)
    entity <- extract_course_entity(resolved_input)
    IF entity IS NULL AND had_pronoun AND session.last_entity NOT NULL THEN
        entity <- session.last_entity
    END IF
    IF entity IS NOT NULL AND tag = "fallback" THEN
        tag <- "courses"        // promote: clear entity beats fallback
    END IF

    // Tier 3 — Knowledge base / response composition
    IF entity AND tag = "fees"    THEN response <- respond_fees_for_course(entity)
    ELSE IF entity AND tag = "courses" THEN response <- respond_course_detail(entity)
    ELSE IF tag IN DYNAMIC_INTENTS THEN response <- build_database_response(tag)
    ELSE                              response <- random_static_template(tag)

    update_session(session, user_input, response, tag, entity)
    log_chat_history(user_input, response, tag, confidence, source)

    RETURN { tag, response, confidence, source, entity }
END
```

---

## 2. Text preprocessing — `clean_text`  &nbsp; *(app/preprocess.py)*

Normalises raw user text into a token sequence the TF-IDF vectoriser
can handle. Uses a hand-built lemmatiser instead of NLTK to keep the
PyInstaller bundle small.

```
ALGORITHM clean_text(text)
INPUT  : text - raw user message
OUTPUT : space-separated string of normalised, lemmatised tokens

BEGIN
    text   <- lowercase(text)
    text   <- regex_replace(text, "[^a-z\s]", "")    // drop punctuation
    text   <- regex_replace(text, "\s+", " ")        // collapse spaces
    tokens <- split(text, " ")

    output <- empty list
    FOR each token IN tokens DO
        token <- simple_lemmatize(token)
        IF token IN KEEP_WORDS OR token NOT IN STOP_WORDS THEN
            output.append(token)
        END IF
    END FOR

    RETURN join(output, " ")
END


ALGORITHM simple_lemmatize(word)
BEGIN
    IF word IN LEMMA_RULES THEN                   // explicit dictionary
        RETURN LEMMA_RULES[word]                  // 'courses' -> 'course'
    END IF
    IF word ends with "ing" AND len(word) > 5 THEN RETURN word[:-3]
    IF word ends with "ed"  AND len(word) > 4 THEN RETURN word[:-2]
    IF word ends with "ies" AND len(word) > 4 THEN RETURN word[:-3] + "y"
    IF word ends with "s"   AND NOT ends with "ss" AND len(word) > 3
                                                  THEN RETURN word[:-1]
    RETURN word
END
```

---

## 3. Intent classification — `predict_intent`  &nbsp; *(app/chat.py)*

Vectorises the cleaned text and asks the trained classifier for the
most likely intent + a calibrated probability.

```
ALGORITHM predict_intent(user_input)
INPUT  : user_input - raw text
OUTPUT : (tag, confidence) where confidence is in [0, 1]

BEGIN
    cleaned   <- clean_text(user_input)
    vector    <- tfidf_vectorizer.transform([cleaned])
    tag       <- model.predict(vector)[0]
    confidence <- max(model.predict_proba(vector)[0])
    RETURN (tag, float(confidence))
END
```

---

## 4. Keyword-rescue safety net — `match_keyword_intent`  &nbsp; *(app/chat.py)*

Second-pass scan that runs only when the SVM's confidence drops
below threshold. Catches phrasings whose distinctive token survived
preprocessing but whose statistical features didn't trigger the
classifier (canonical example: *"What events are coming up?"*).

```
ALGORITHM match_keyword_intent(text)
INPUT  : text  - resolved user input
OUTPUT : intent tag, OR NULL if no keyword fires

BEGIN
    text_lo <- lowercase(text)

    // INTENT_KEYWORDS is an ordered tuple - more specific intents
    // (events, hostel, scholarship) come first so 'mba' beats
    // 'admission' in compound phrases.
    FOR each (intent_tag, keyword_list) IN INTENT_KEYWORDS DO
        FOR each kw IN keyword_list DO
            IF kw contains a space THEN
                IF kw is a substring of text_lo THEN
                    RETURN intent_tag
                END IF
            ELSE
                IF word_boundary_match(text_lo, kw) THEN
                    RETURN intent_tag
                END IF
            END IF
        END FOR
    END FOR

    RETURN NULL
END
```

---

## 5. Anaphora resolution — `resolve_pronouns`  &nbsp; *(app/context.py)*

Substitutes pronouns (`it`, `this course`, `that one`, `the
programme`) with the entity remembered from earlier turns. Lets the
bot answer follow-ups like *"price of it"* after *"tell me about
MBA"*.

```
ALGORITHM resolve_pronouns(text, session)
INPUT  : text     - user message
         session  - per-session state dict (or NULL)
OUTPUT : text with pronouns substituted, or original text unchanged

BEGIN
    IF session IS NULL OR session.last_entity IS NULL THEN
        RETURN text
    END IF

    entity <- session.last_entity
    resolved <- text

    FOR each pattern IN PRONOUN_PATTERNS DO
        // PRONOUN_PATTERNS = ['this course', 'that course',
        //                     'the course',  'this program',
        //                     'this one',    'that one', 'it', ...]
        resolved <- regex_replace_word_boundary(resolved, pattern, entity)
    END FOR

    RETURN resolved
END
```

---

## 6. Entity extraction — `extract_course_entity`  &nbsp; *(app/chat.py)*

Spots a course mention in the user's message and returns the
canonical course name. Used to route per-entity responses (per-course
detail, per-course fee). Three-pass match: full name → course code →
alias dictionary.

```
ALGORITHM extract_course_entity(text)
INPUT  : text - resolved user message
OUTPUT : canonical course name, OR NULL

BEGIN
    rows    <- db.list_courses()
    text_lo <- lowercase(text)

    // Pass 1: full canonical name match ('BSc Computer Science')
    FOR each row IN rows DO
        IF lowercase(row.name) IS substring of text_lo THEN
            RETURN row.name
        END IF
    END FOR

    // Pass 2: course code match ('CS-BSC', 'MBA')
    FOR each row IN rows DO
        IF word_boundary_match(text_lo, lowercase(row.code)) THEN
            RETURN row.name
        END IF
    END FOR

    // Pass 3: alias match ('cs', 'mba', 'data science')
    FOR each (alias, needle) IN COURSE_ALIASES DO
        IF word_boundary_match(text_lo, alias) THEN
            FOR each row IN rows DO
                IF needle is substring of lowercase(row.name) THEN
                    RETURN row.name
                END IF
            END FOR
        END IF
    END FOR

    RETURN NULL
END
```

---

## 7. Training pipeline — `train_and_evaluate`  &nbsp; *(app/train.py)*

3-model bake-off picked by 5-fold cross-validation. Runs at initial
seed time and on every auto-retrain.

```
ALGORITHM train_and_evaluate()
OUTPUT : name of the winning model

BEGIN
    // Step 1: load data
    intents      <- load_json("data/intents.json")
    learned_rows <- db.get_learned_patterns(approved_only = TRUE)

    // Step 2: build training corpus
    patterns <- empty list
    tags     <- empty list
    FOR each intent IN intents DO
        FOR each pattern IN intent.patterns DO
            patterns.append(clean_text(pattern))
            tags.append(intent.tag)
        END FOR
    END FOR
    FOR each row IN learned_rows DO
        patterns.append(clean_text(row.pattern))
        tags.append(row.intent)
    END FOR

    // Step 3: vectorise
    vectorizer <- TfidfVectorizer(max_features = 500)
    X          <- vectorizer.fit_transform(patterns)
    y          <- array(tags)

    // Step 4: stratified 80/20 split
    X_train, X_test, y_train, y_test <-
        train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)

    // Step 5: bake-off
    candidates <- {
        "Naive Bayes":   MultinomialNB(),
        "SVM":           SVC(kernel = "linear", probability = TRUE),
        "Random Forest": RandomForestClassifier(n_estimators = 100)
    }

    best_name <- NULL
    best_cv   <- 0
    FOR each (name, model) IN candidates DO
        model.fit(X_train, y_train)
        cv_scores <- cross_val_score(model, X, y, cv = 5)
        IF mean(cv_scores) > best_cv THEN
            best_cv   <- mean(cv_scores)
            best_name <- name
        END IF
    END FOR

    // Step 6: persist + cleanup
    pickle.dump(candidates[best_name], "models/chatbot_model.pkl")
    pickle.dump(vectorizer,            "models/vectorizer.pkl")
    write_model_info_txt(best_name, train_acc, test_acc, cv_mean,
                         len(patterns), len(intents))
    db.mark_patterns_used()

    RETURN best_name
END
```

---

## 8. Feedback recording — `record_feedback`  &nbsp; *(app/learning.py)*

End-user thumbs-up / thumbs-down. Thumbs-down with a suggested intent
goes into the admin review queue (Tier 1, approved = 0). It does
**not** retrain the model.

```
ALGORITHM record_feedback(message, response, predicted, conf,
                          helpful, expected)
OUTPUT : { feedback_id, retrained = FALSE, pending_review_count }

BEGIN
    feedback_id <- db.log_feedback(message, response, predicted,
                                    conf, helpful, expected)

    IF helpful = FALSE AND expected IS NOT NULL THEN
        db.add_learned_pattern(
            pattern   = message,
            intent    = expected,
            source    = "feedback_correction",
            approved  = FALSE       // <- waits for admin review
        )
    END IF

    RETURN {
        feedback_id          : feedback_id,
        retrained            : FALSE,
        pending_review_count : db.count_pending_review()
    }
END
```

---

## 9. Auto-retrain trigger — `maybe_auto_retrain`  &nbsp; *(app/learning.py)*

Runs after every admin approve / direct-teach. Triggers a full
retrain once 5 approved-but-unused patterns have accumulated.

```
CONSTANT AUTO_RETRAIN_THRESHOLD = 5

ALGORITHM maybe_auto_retrain()
OUTPUT : TRUE if a retrain happened, FALSE otherwise

BEGIN
    pending_count <- db.count_pending_patterns()
    IF pending_count >= AUTO_RETRAIN_THRESHOLD THEN
        train_and_evaluate()
        // (Flask hot-swaps EduBot() in app.py after this returns)
        RETURN TRUE
    END IF
    RETURN FALSE
END
```

---

## 10. Input quality validation — `check_message_quality`  &nbsp; *(app/validate.py)*

Three-rule heuristic gate that rejects keyboard mashing and
phone-number-style numeric spam without blocking emails, prices or
years. Server-side gate on `/chat`; mirrored client-side for instant
feedback.

```
CONSTANT MAX_ALPHA_RUN     = 30      // 30+ unbroken letters = gibberish
CONSTANT MAX_DIGIT_RUN     = 9       // 10+ unbroken digits = phone-like
CONSTANT MIN_LETTER_RATIO  = 0.30    // < 30% letters = mostly symbols

ALGORITHM check_message_quality(text)
OUTPUT : nothing on success; raises ValidationError otherwise

BEGIN
    // Rule 1: gibberish (one impossibly-long letter run)
    IF regex_match(text, "[A-Za-z]{31,}") THEN
        RAISE ValidationError("looks like gibberish")
    END IF

    // Rule 2: phone-number-like digit spam
    IF regex_match(text, "[0-9]{10,}") THEN
        RAISE ValidationError("long number sequences not allowed")
    END IF

    // Rule 3: too few letters (only enforced for messages >= 4 chars
    //         so '?' and 'ok' still pass)
    IF length(text) >= 4 THEN
        letter_count <- count_letters(text)
        IF letter_count / length(text) < MIN_LETTER_RATIO THEN
            RAISE ValidationError("too few letters - ask in words")
        END IF
    END IF
END
```

---

## 11. Auto-seed on first boot — `cold_start`  &nbsp; *(app.py)*

Idempotent server boot. Creates the SQLite schema, seeds the
knowledge base on a first / fresh deploy, then loads the trained
model into memory.

```
ALGORITHM cold_start()
BEGIN
    db.init_schema()                          // idempotent CREATE TABLE IF NOT EXISTS

    IF db.stats().courses = 0 THEN            // empty knowledge base?
        log("[boot] empty DB - seeding")
        seed_db.seed_all()                    // populates 7 read-only tables
    END IF

    bot <- EduBot()                            // loads pickle + opens DB
    log("EduBot ready")
END
```

---

## 12. Two-tier learning loop (high-level)

The system-level orchestration of how a single user's thumbs-down
becomes a model improvement. This is the algorithm the marker is
testing under "indication of machine learning - chatbot updating its
own knowledge base" (5 marks).

```
ALGORITHM two_tier_learning_lifecycle()
ACTORS : End-User, Admin, Application

BEGIN
    // ============ TIER 1 - END USER (open) ============
    End-User asks a question
        -> Application: POST /chat -> get_response()
        -> reply rendered with thumbs buttons

    IF End-User clicks thumbs-down + picks a correct intent THEN
        -> Application: POST /feedback -> record_feedback(helpful = FALSE,
                                                          expected = X)
        -> db.add_learned_pattern(approved = FALSE)
        // Pattern is now PENDING REVIEW. NOT used for training yet.
    END IF

    // ============ TIER 2 - ADMIN (gated) ===========
    Admin opens /admin and sees the pending suggestion

    IF Admin clicks "Approve" THEN
        -> Application: POST /admin/approve/<id>
                         -> db.approve_pattern(id)         // approved = 1
                         -> maybe_auto_retrain()
                         -> IF threshold met:
                              -> train_and_evaluate()
                              -> Flask hot-swaps EduBot()
                              -> NEW MODEL IS LIVE

    ELSE IF Admin clicks "Discard" THEN
        -> db.discard_pattern(id)                          // row deleted

    ELSE IF Admin uses Teach form directly THEN
        -> db.add_learned_pattern(approved = TRUE)         // pre-approved
        -> maybe_auto_retrain()
    END IF
END
```

The trust split (end users can suggest, only admins can approve) is
enforced at three layers — HTTP (Basic Auth on `/admin`), application
(`record_feedback` writes `approved = FALSE`), and database
(`get_learned_patterns(approved_only = TRUE)` is the only loader).

---

## How to use this file

For the **technical documentation** (page 9 marking scheme, *"Algorithms
(e.g. flow charts, pseudo codes) — 2 marks"*):

- Embed **Algorithms 1, 2, 7, and 12** at minimum. They cover the main
  chat flow, preprocessing, training, and the feedback loop — the
  four most marker-relevant algorithms.
- Add Algorithms 3, 4, 5, 8, 9, 10 if you have space — they show
  depth.
- Algorithms 6 and 11 are nice-to-have but less critical for marks.

For the **viva**, having all 12 in one file means you can flip to any
of them when a panel question lands on that part of the system.
