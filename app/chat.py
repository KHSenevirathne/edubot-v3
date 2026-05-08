import json
import pickle
import random
import re
import os
import sys

# Local imports - sys.path tweak so direct execution still works.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import clean_text  # noqa: E402
import database as db              # noqa: E402
import context as ctx              # noqa: E402


# Intents whose responses depend on data that can change over time.
# Live from the SQLite DB.
DYNAMIC_INTENTS = {
    'courses', 'fees', 'admission', 'scholarship', 'exams',
    'timetable', 'library', 'contact', 'faculty', 'hostel', 'events'
}


# Course-name aliases the user might type.
_COURSE_ALIASES = {
    'cs':                 'computer science',
    'comp sci':           'computer science',
    'computer science':   'computer science',
    'it':                 'information technology',
    'information technology': 'information technology',
    'software engineering':   'software engineering',
    'software eng':           'software engineering',
    'data science':           'data science',
    'cyber security':         'cyber security',
    'cybersecurity':          'cyber security',
    'mba':                'mba',
    'bba':                'bba',
    'business':           'business administration',
    'business administration': 'business administration',
}


# Keyword safety net for the classifier.
_INTENT_KEYWORDS = (
    ('events',      ('event', 'events', 'hackathon', 'hackathons',
                     'fest', 'festival', 'festivals',
                     'seminar', 'seminars', 'workshop', 'workshops',
                     'club', 'clubs', 'society', 'societies')),
    ('hostel',      ('hostel', 'dorm', 'dormitory', 'accommodation',
                     'lodging', 'room and board')),
    ('library',     ('library',)),
    ('scholarship', ('scholarship', 'scholarships', 'bursary', 'fee waiver',
                     'financial aid', 'grant', 'grants')),
    ('faculty',     ('lecturer', 'lecturers', 'professor', 'professors',
                     'faculty', 'dean', 'teacher', 'teachers')),
    ('exams',       ('exam', 'exams', 'examination', 'midterm', 'midterms',
                     'finals', 'resit')),
    ('timetable',   ('timetable', 'schedule', 'class hours',
                     'lecture schedule')),
    ('admission',   ('admission', 'admissions', 'apply', 'application',
                     'enrol', 'enroll', 'enrolment', 'enrollment')),
    ('fees',        ('tuition', 'fees', 'fee', 'price', 'cost')),
    ('contact',     ('contact', 'phone number', 'email address',
                     'helpline')),
    ('courses',     ('course', 'courses', 'programme', 'programmes',
                     'program', 'programs', 'degree', 'degrees',
                     'major', 'majors')),
)


def _match_keyword_intent(user_input):
    """Scan user_input for an unambiguous intent keyword.

    Returns the intent tag, or None when no keyword fires. Matching is
    word-boundary based on the lower-cased input so 'event' inside
    'eventually' or 'fee' inside 'feedback' don't trigger a false hit.
    """
    if not user_input:
        return None
    text_lo = user_input.lower()
    for tag, keywords in _INTENT_KEYWORDS:
        for kw in keywords:
            if ' ' in kw:
                if kw in text_lo:
                    return tag
            else:
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lo):
                    return tag
    return None


class EduBot:
    """Main chatbot class. Loads model, predicts intent, builds response."""

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', 'chatbot_model.pkl')
        vec_path = os.path.join(base_dir, 'models', 'vectorizer.pkl')
        intents_path = os.path.join(base_dir, 'data', 'intents.json')

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Model not found. Run `python app/train.py` first."
            )

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(vec_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(intents_path, 'r', encoding='utf-8') as f:
            self.intents_data = json.load(f)

        # tag -> list[response_template]
        self.response_map = {
            intent['tag']: intent['responses']
            for intent in self.intents_data['intents']
        }

        # Below this confidence, fall back instead of trusting the prediction.
        self.confidence_threshold = 0.4

        # Make sure the DB exists - cheap, idempotent.
        db.init_schema()

    # ---------------- Inference ----------------

    def predict_intent(self, user_input):
        """Return (predicted_tag, confidence)."""
        cleaned = clean_text(user_input)
        vec = self.vectorizer.transform([cleaned])
        tag = self.model.predict(vec)[0]

        # SVM with probability=True / NB / RF all expose predict_proba.
        if hasattr(self.model, 'predict_proba'):
            confidence = float(max(self.model.predict_proba(vec)[0]))
        else:
            confidence = 1.0
        return tag, confidence

    def get_response(self, user_input, session=None):
        """Public entry point.

        Returns dict {response, tag, confidence, source, entity}.

        When `session` is provided (a dict from context.get_session), the
        bot will:
          - resolve pronouns ("it", "this course") to the entity it
            remembered from earlier turns,
          - update that memory with any new entity it spots in the
            current message,
          - produce per-entity answers (e.g. fees for ONE course rather
            than the full price list) when the user clearly means a
            specific programme.
        """
        if not user_input or not user_input.strip():
            return {
                'tag': 'fallback',
                'response': "Please type a question and I'll try to help!",
                'confidence': 0.0,
                'source': 'static',
                'entity': None,
            }

        # ---- Dialogue management: anaphora + entity tracking ----
        had_pronoun = ctx.has_pronoun(user_input)
        resolved_input = ctx.resolve_pronouns(user_input, session) if session else user_input

        # If the user used a pronoun but we don't have anything to point
        # it at, ask which programme they mean rather than guessing.
        if (
            session is not None
            and had_pronoun
            and not session.get('last_entity')
        ):
            response = (
                "Which programme would you like to know about? "
                "For example: 'BSc Computer Science', 'BSc Data Science', or 'MBA'."
            )
            db.log_chat(user_input, response, 'clarify', 1.0, 'static')
            return {
                'tag': 'clarify',
                'response': response,
                'confidence': 1.0,
                'source': 'static',
                'entity': None,
            }

        tag, confidence = self.predict_intent(resolved_input)

        # Low-confidence rescue. Before forcing fallback, see whether
        # the message contains an unambiguous intent keyword
        if confidence < self.confidence_threshold:
            keyword_tag = _match_keyword_intent(resolved_input)
            if keyword_tag:
                tag = keyword_tag
                confidence = max(confidence, 0.6)
            else:
                tag = 'fallback'

        # carry the previous entity forward.
        entity = self._extract_course_entity(resolved_input)
        if entity is None and session and session.get('last_entity') and had_pronoun:
            entity = session['last_entity']

        # If the user clearly named a course, refuse to silently fall
        # back - the entity itself is enough signal that they want the
        # programme detail.
        if entity and tag == 'fallback':
            tag = 'courses'

        response, source = self._build_response(tag, entity=entity)

        # Persist for future turns and analytics.
        if session is not None:
            ctx.update_session(
                session,
                user_message=user_input,
                bot_response=response,
                intent=tag,
                entity=entity,
            )
        db.log_chat(user_input, response, tag, confidence, source)

        return {
            'tag': tag,
            'response': response,
            'confidence': round(confidence, 3),
            'source': source,
            'entity': entity,
        }

    # ---------------- Response composition ----------------

    def _build_response(self, tag, entity=None):
        """Return (response_text, source_label).

        source_label is one of: 'database', 'static', 'fallback'.
        When `entity` (a canonical course name) is set, the response is
        narrowed to that single programme - this is how multi-turn
        consulting works ("price of it" -> price of THE course just
        discussed)."""

        # Per-entity overrides come first - they only fire when the
        # user clearly meant a specific programme.
        if entity:
            if tag == 'fees':
                resp = self._respond_fees_for_course(entity)
                if resp:
                    return resp, 'database'
            if tag == 'courses':
                resp = self._respond_course_detail(entity)
                if resp:
                    return resp, 'database'

        if tag in DYNAMIC_INTENTS:
            try:
                response = self._db_response(tag)
                if response:
                    return response, 'database'
            except Exception as e:
                # Don't crash the chat session if a DB query fails -
                # degrade gracefully to the JSON template.
                print(f"[chat] DB lookup failed for tag={tag}: {e}")

        # Static path: random pick from intents.json templates.
        if tag in self.response_map:
            return random.choice(self.response_map[tag]), 'static'

        return ("I'm sorry, I don't have information about that. "
                "Please try a different question."), 'fallback'

    # ---------------- Entity extraction (for multi-turn) ----------------

    @staticmethod
    def _extract_course_entity(text):
        """Identify a specific course mentioned in `text`.

        Strategy:
          1. Direct substring match against canonical course names from
             the DB (highest precision).
          2. Course-code match (e.g. 'CS101').
          3. Alias match against _COURSE_ALIASES for shorthand the user
             is likely to type ('cs', 'mba', 'data science').

        Returns the canonical course name, or None if no match.
        """
        if not text:
            return None
        text_lo = text.lower()
        rows = db.list_courses()
        if not rows:
            return None

        # 1) full canonical name appears verbatim
        for r in rows:
            if r['name'] and r['name'].lower() in text_lo:
                return r['name']
        # 2) course code (CS101, MBA01, ...)
        for r in rows:
            code = r.get('code') or ''
            if code and re.search(r'\b' + re.escape(code.lower()) + r'\b', text_lo):
                return r['name']
        # 3) alias -> needle, then needle must appear in some course name
        for alias, needle in _COURSE_ALIASES.items():
            if not re.search(r'\b' + re.escape(alias) + r'\b', text_lo):
                continue
            for r in rows:
                if needle in (r['name'] or '').lower():
                    return r['name']
        return None

    # The following _* methods build live responses from the DB.
    # Each one corresponds to one DYNAMIC_INTENTS entry.

    def _db_response(self, tag):
        builder = {
            'courses':     self._respond_courses,
            'fees':        self._respond_fees,
            'admission':   self._respond_admission,
            'scholarship': self._respond_scholarship,
            'exams':       self._respond_exams,
            'timetable':   self._respond_timetable,
            'library':     self._respond_library,
            'contact':     self._respond_contact,
            'faculty':     self._respond_faculty,
            'hostel':      self._respond_hostel,
            'events':      self._respond_events,
        }.get(tag)
        return builder() if builder else None

    @staticmethod
    def _respond_courses():
        rows = db.list_courses()
        if not rows:
            return None
        ug = [r for r in rows if r['level'] == 'Undergraduate']
        pg = [r for r in rows if r['level'] == 'Postgraduate']
        lines = ["Here are our current programmes:"]
        if ug:
            lines.append("\nUndergraduate:")
            for r in ug:
                lines.append(f"  - {r['name']} ({r['code']}) "
                             f"- {r['faculty']} - ${r['fee_per_year']}/year")
        if pg:
            lines.append("\nPostgraduate:")
            for r in pg:
                lines.append(f"  - {r['name']} ({r['code']}) "
                             f"- {r['faculty']} - ${r['fee_per_year']}/year")
        return "\n".join(lines)

    @staticmethod
    def _respond_course_detail(course_name):
        """Per-course detail card (used when the user has narrowed down to
        a single programme during a multi-turn chat)."""
        rows = [r for r in db.list_courses() if r['name'] == course_name]
        if not rows:
            return None
        r = rows[0]
        lines = [f"Here are the details for {r['name']} ({r['code']}):"]
        lines.append(f"  - Level:    {r['level']}")
        lines.append(f"  - Faculty:  {r['faculty']}")
        lines.append(f"  - Duration: {r['duration_years']} years")
        lines.append(f"  - Tuition:  ${r['fee_per_year']}/year")
        if r.get('description'):
            lines.append("")
            lines.append(r['description'])
        return "\n".join(lines)

    @staticmethod
    def _respond_fees_for_course(course_name):
        """Fees for ONE specific programme - the answer to a follow-up
        like 'price of it' once a course is in the conversation context."""
        rows = [r for r in db.list_courses() if r['name'] == course_name]
        if not rows:
            return None
        r = rows[0]
        per_sem = r['fee_per_year'] // 2
        try:
            total = int(round(r['fee_per_year'] * float(r['duration_years'])))
        except (TypeError, ValueError):
            total = None
        lines = [
            f"Tuition for {r['name']} ({r['code']}):",
            f"  - ${r['fee_per_year']}/year",
            f"  - ${per_sem}/semester (paid in two instalments)",
        ]
        if total is not None:
            lines.append(f"  - Total programme cost: ~${total} "
                         f"over {r['duration_years']} years")
        return "\n".join(lines)

    @staticmethod
    def _respond_fees():
        rows = db.list_courses()
        if not rows:
            return None
        lines = ["Here is our current fee structure (per year):"]
        seen_levels = set()
        # Sort so undergrads come first, then postgrads
        for r in sorted(rows, key=lambda r: (r['level'], r['fee_per_year'])):
            seen_levels.add(r['level'])
            lines.append(f"  - {r['name']}: ${r['fee_per_year']}/year")
        lines.append("\nFees may be paid per semester. "
                     "Scholarships and instalment plans are available - "
                     "ask about scholarships if you'd like to know more.")
        return "\n".join(lines)

    @staticmethod
    def _respond_admission():
        facts = db.get_facts_by_category('admission')
        if not facts:
            return None
        lines = [
            "How to apply:",
            f"  1. Visit our online portal: {facts.get('admission_portal', 'university.edu/apply')}",
            "  2. Create an account and fill in your details",
            f"  3. Upload required documents - {facts.get('admission_documents', 'ID, transcripts, statement')}",
            f"  4. Pay the application fee - {facts.get('admission_fee', '$25')}",
            "  5. Submit and wait for confirmation",
        ]
        if facts.get('admission_deadline'):
            lines.append(f"\nDeadlines: {facts['admission_deadline']}")
        if facts.get('admission_response'):
            lines.append(f"Decisions are issued: {facts['admission_response']}")
        return "\n".join(lines)

    @staticmethod
    def _respond_scholarship():
        rows = db.list_scholarships()
        if not rows:
            return None
        lines = ["We offer the following scholarships and financial aid:"]
        for r in rows:
            lines.append(f"  - {r['name']} (up to {r['max_percentage']}% off) "
                         f"- {r['eligibility']}")
        lines.append("\nApply through the student portal under 'Financial Aid'.")
        return "\n".join(lines)

    @staticmethod
    def _respond_exams():
        rows = db.list_exams()
        if not rows:
            return None
        lines = ["Upcoming exam schedule:"]
        for r in rows:
            window = (f"{r['start_date']} to {r['end_date']}"
                      if r['start_date'] != r['end_date'] else r['start_date'])
            lines.append(f"  - {r['exam_type']}: {window} "
                         f"({r['format']})")
        grading = db.get_fact('grading_scale')
        if grading:
            lines.append(f"\nGrading: {grading}")
        return "\n".join(lines)

    @staticmethod
    def _respond_timetable():
        facts = db.get_facts_by_category('timetable')
        if not facts:
            return None
        lines = [facts.get('timetable_general', '')]
        if facts.get('timetable_portal'):
            lines.append(facts['timetable_portal'])
        if facts.get('timetable_lecture_len'):
            lines.append(facts['timetable_lecture_len'])
        return "\n".join(line for line in lines if line)

    @staticmethod
    def _respond_library():
        facts = db.get_facts_by_category('library')
        if not facts:
            return None
        lines = ["Library information:"]
        if facts.get('library_location'):
            lines.append(f"  - Location: {facts['library_location']}")
        if facts.get('library_hours_weekday'):
            lines.append(f"  - {facts['library_hours_weekday']}")
        if facts.get('library_hours_weekend'):
            lines.append(f"  - {facts['library_hours_weekend']}")
        if facts.get('library_borrow_limit'):
            lines.append(f"  - {facts['library_borrow_limit']}")
        if facts.get('library_digital'):
            lines.append(f"  - {facts['library_digital']}")
        return "\n".join(lines)

    @staticmethod
    def _respond_contact():
        facts = db.get_facts_by_category('contact')
        if not facts:
            return None
        lines = ["You can reach the university through:"]
        for label, key in [
            ("Email",      'contact_email'),
            ("Phone",      'contact_phone'),
            ("Address",    'contact_address'),
            ("Admin office", 'contact_admin_office'),
            ("Student support", 'contact_support'),
            ("Emergency",  'contact_emergency'),
        ]:
            if facts.get(key):
                lines.append(f"  - {label}: {facts[key]}")
        return "\n".join(lines)

    @staticmethod
    def _respond_faculty():
        rows = db.list_faculty()
        if not rows:
            return None
        lines = ["Our faculty members include:"]
        for r in rows:
            tag = " (Dean)" if r['is_dean'] else ""
            line = f"  - {r['title']} {r['name']}{tag} - {r['department']}"
            if r['expertise']:
                line += f" ({r['expertise']})"
            lines.append(line)
            if r['office_hours']:
                lines.append(f"      Office hours: {r['office_hours']}")
        return "\n".join(lines)

    @staticmethod
    def _respond_hostel():
        rows = db.list_hostel_rooms()
        if not rows:
            return None
        lines = ["On-campus accommodation options:"]
        for r in rows:
            lines.append(f"  - {r['room_type']} (sleeps {r['capacity']}): "
                         f"${r['price_per_semester']}/semester")
            if r['amenities']:
                lines.append(f"      Amenities: {r['amenities']}")
        lines.append("\nApply via the student portal under 'Accommodation'. "
                     "Spots are limited - apply early!")
        return "\n".join(lines)

    @staticmethod
    def _respond_events():
        rows = db.list_events()
        if not rows:
            return None
        lines = ["Upcoming events:"]
        for r in rows:
            window = (f"{r['start_date']} to {r['end_date']}"
                      if r['end_date'] and r['start_date'] != r['end_date']
                      else r['start_date'])
            lines.append(f"  - {r['name']} ({r['category']}): "
                         f"{window} @ {r['location']}")
        return "\n".join(lines)


# Interactive terminal mode - useful for testing without the web server.
if __name__ == "__main__":
    print("=" * 50)
    print("  EduBot v3 - Terminal Chat (Testing Mode)")
    print("  Type 'quit' to exit")
    print("=" * 50)

    bot = EduBot()
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {'quit', 'exit', 'q'}:
            print("Bot: Goodbye!")
            break
        result = bot.get_response(user_input)
        print(f"Bot ({result['source']}): {result['response']}")
        print(f"     [intent={result['tag']} | "
              f"confidence={result['confidence']:.1%}]")
