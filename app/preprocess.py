"""
preprocess.py - Text Preprocessing Module for EduBot v3
Handles tokenization, lemmatization, and text cleaning for the NLP pipeline.

Implements a self-contained preprocessing layer (no NLTK data downloads
required) so the packaged executable can run on a clean machine without
network access.
"""

import re

# English stopwords list. Hand-tuned: words that carry intent signal
# (how, what, when, etc.) are deliberately KEPT.
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'between', 'out', 'off', 'over', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'all', 'each',
    'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'just', 'because', 'until', 'while', 'about', 'against', 'up', 'down',
    'been', 'being', 'have', 'has', 'had', 'having', 'was', 'were', 'be',
    'am', 'did', 'doing', 'would', 'should', 'could', 'ought', 'might',
    'shall', 'will', 'also', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'he', 'him', 'his', 'himself', 'she', 'her',
    'hers', 'herself', 'we', 'us', 'our', 'ours', 'ourselves', 'you',
    'your', 'yours', 'yourself', 'yourselves', 'this', 'that', 'these',
    'those', 'whom', 'if', 'else',
    'any', 'much', 'now', 'ever', 'well', 'back', 'even', 'still',
    'take', 'since', 'another', 'however', 'two', 'like', 'go',
    'see', 'get', 'got', 'really', 'right', 'think',
    'come', 'good', 'look', 'thing', 'use',
}

# Whitelist: question/intent words that must survive even if they
# overlap with the stopwords list above.
KEEP_WORDS = {'how', 'what', 'when', 'where', 'who', 'which', 'can', 'do',
              'does', 'is', 'are', 'my', 'i', 'me', 'need', 'want', 'tell',
              'show', 'help', 'way'}

# Lemmatisation lookup: irregular plurals + common verb conjugations.
# Falls back to suffix stripping for words not listed.
LEMMA_RULES = {
    'courses': 'course', 'programs': 'program', 'programmes': 'program',
    'degrees': 'degree', 'fees': 'fee', 'exams': 'exam',
    'examinations': 'exam', 'examination': 'exam',
    'classes': 'class', 'lectures': 'lecture', 'books': 'book',
    'students': 'student', 'teachers': 'teacher', 'lecturers': 'lecturer',
    'professors': 'professor', 'scholarships': 'scholarship',
    'events': 'event', 'activities': 'activity', 'facilities': 'facility',
    'rooms': 'room', 'hours': 'hour', 'dates': 'date',
    'applying': 'apply', 'applied': 'apply', 'applies': 'apply',
    'studying': 'study', 'studied': 'study', 'studies': 'study',
    'offering': 'offer', 'offered': 'offer', 'offers': 'offer',
    'starting': 'start', 'started': 'start', 'starts': 'start',
    'borrowing': 'borrow', 'borrowed': 'borrow', 'borrows': 'borrow',
    'payments': 'payment', 'paying': 'pay', 'paid': 'pay',
    'enrolling': 'enroll', 'enrolled': 'enroll', 'enrollment': 'enroll',
    'registering': 'register', 'registered': 'register',
    'opening': 'open', 'opened': 'open', 'opens': 'open',
    'closing': 'close', 'closed': 'close', 'closes': 'close',
    'departments': 'department', 'faculties': 'faculty',
    'hostels': 'hostel', 'dormitories': 'dormitory', 'dorms': 'dorm',
    'libraries': 'library', 'workshops': 'workshop',
    'schedules': 'schedule', 'timetables': 'timetable',
    'results': 'result', 'grades': 'grade', 'marks': 'mark',
    'available': 'available', 'information': 'information',
}


def simple_lemmatize(word):
    """Return the dictionary form of a single word.

    Strategy: lookup table first, then suffix stripping (-ing, -ed, -ies, -s).
    Words shorter than the rule's minimum length are returned unchanged so
    we don't strip 'is' or 'as' down to nothing.
    """
    if word in LEMMA_RULES:
        return LEMMA_RULES[word]

    if word.endswith('ing') and len(word) > 5:
        base = word[:-3]
        if len(base) > 2:
            return base
    if word.endswith('ed') and len(word) > 4:
        base = word[:-2]
        if len(base) > 2:
            return base
    if word.endswith('ies') and len(word) > 4:
        return word[:-3] + 'y'
    if word.endswith('s') and not word.endswith('ss') and len(word) > 3:
        return word[:-1]

    return word


def clean_text(text):
    """Normalise a raw user message into a token sequence ready for TF-IDF.

    Pipeline:
      1. Lower-case.
      2. Strip everything that is not a letter or whitespace.
      3. Tokenise on whitespace.
      4. Lemmatise each token.
      5. Drop stopwords (but keep question words on the whitelist).
      6. Re-join into a single space-separated string.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()

    cleaned_words = []
    for word in words:
        word = simple_lemmatize(word)
        if word not in STOP_WORDS or word in KEEP_WORDS:
            cleaned_words.append(word)

    return ' '.join(cleaned_words)


def preprocess_patterns(patterns):
    """Apply clean_text to every pattern in a list."""
    return [clean_text(pattern) for pattern in patterns]


# Smoke test - verifies that clean_text strips punctuation, lemmatises
# plurals, and keeps interrogatives.
if __name__ == "__main__":
    test_sentences = [
        "How do I apply for admission?",
        "What courses do you offer!!!",
        "I want to know about the LIBRARY hours",
        "Tell me about scholarships & financial aid",
        "When are the exams starting???",
        "Is hostel available for students?",
        "Who are the professors in CS department?"
    ]

    print("=" * 60)
    print("  PREPROCESSING TEST")
    print("=" * 60)
    for sentence in test_sentences:
        cleaned = clean_text(sentence)
        print(f"  Original : {sentence}")
        print(f"  Cleaned  : {cleaned}")
        print("-" * 60)
