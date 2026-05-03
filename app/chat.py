"""
chat.py - Inference Engine for EduBot v3

Three-tier responsibility (per the assignment brief):
  - Loads the trained intent classifier (TIER 2: Inference Engine)
  - Routes "dynamic" intents through SQLite (TIER 3: Database)
  - Falls back to intents.json templates for "static" small-talk
  - Logs every turn to chat_history for analytics

The DYNAMIC_INTENTS set is the single source of truth that decides
whether a response is built from live DB data or from a stock template.
"""

import json
import pickle
import random
import os
import sys

# Local imports - sys.path tweak so direct execution still works.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import clean_text  # noqa: E402
import database as db              # noqa: E402


# Intents whose responses depend on data that can change over time.
# These get composed live from the SQLite DB.
DYNAMIC_INTENTS = {
    'courses', 'fees', 'admission', 'scholarship', 'exams',
    'timetable', 'library', 'contact', 'faculty', 'hostel', 'events'
}


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

    def get_response(self, user_input):
        """Public entry point. Returns dict {response, tag, confidence, source}."""
        if not user_input or not user_input.strip():
            return {
                'tag': 'fallback',
                'response': "Please type a question and I'll try to help!",
                'confidence': 0.0,
                'source': 'static',
            }

        tag, confidence = self.predict_intent(user_input)

        # Low-confidence answers go straight to fallback.
        if confidence < self.confidence_threshold:
            tag = 'fallback'

        response, source = self._build_response(tag)

        # Log to chat_history (analytics + test plan evidence).
        db.log_chat(user_input, response, tag, confidence, source)

        return {
            'tag': tag,
            'response': response,
            'confidence': round(confidence, 3),
            'source': source,
        }

    # ---------------- Response composition ----------------

    def _build_response(self, tag):
        """Return (response_text, source_label).

        source_label is one of: 'database', 'static', 'fallback'.
        Used by the test plan to verify the three-tier architecture."""

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
        lines.append("\nWould you like details about any specific programme?")
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
