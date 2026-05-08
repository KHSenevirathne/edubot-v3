import re

# Maximum length for any single text field.
MAX_MESSAGE_LEN = 500
MAX_PATTERN_LEN = 200
MIN_TEXT_LEN = 1

# Longest unbroken alphabetic run we'll allow inside a single token.
MAX_ALPHA_RUN = 30

# Longest unbroken digit run we'll allow.
MAX_DIGIT_RUN = 15

# Minimum proportion of letters in the message.
# (>30% letters).
MIN_LETTER_RATIO = 0.30

_ALPHA_RUN_RE = re.compile(r'[A-Za-z]{' + str(MAX_ALPHA_RUN + 1) + r',}')
_DIGIT_RUN_RE = re.compile(r'\d{' + str(MAX_DIGIT_RUN + 1) + r',}')
_LETTER_RE    = re.compile(r'[A-Za-z]')
_PHONE_LIKE_RE = re.compile(r'^[\d\s+\-()]+$')

# Strip ASCII control codes (NUL, BEL, VT, FF, etc.) but keep newlines
_BAD_CHARS = re.compile(
    '['
    '\x00-\x08\x0b\x0c\x0e-\x1f\x7f'   # ASCII control codes
    '​-‏'                    # zero-width / direction
    '  '                     # line / paragraph separators
    '﻿'                           # byte-order mark
    ']'
)


class ValidationError(ValueError):
    """Raised when a field fails validation. Carries the offending
    field name so the API response can point the caller at it."""

    def __init__(self, field, message):
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message


def clean_text_field(value, field, min_len=MIN_TEXT_LEN, max_len=MAX_MESSAGE_LEN,
                     check_quality=False):
    """Validate + normalise a free-text field.

    Steps:
      1. Reject None / non-string types.
      2. Strip leading and trailing whitespace.
      3. Drop control characters and zero-width chars.
      4. Enforce min/max length.
      5. (chat messages only, when check_quality=True) reject long
         gibberish runs, long digit runs, and inputs with too few
         letters.
    """
    if value is None:
        raise ValidationError(field, 'is required')
    if not isinstance(value, str):
        raise ValidationError(field, 'must be a string')

    cleaned = _BAD_CHARS.sub('', value).strip()

    if len(cleaned) < min_len:
        raise ValidationError(
            field, f'must be at least {min_len} character(s) after trim'
        )
    if len(cleaned) > max_len:
        raise ValidationError(
            field, f'must be at most {max_len} characters'
        )
    if check_quality:
        check_message_quality(cleaned, field)
    return cleaned


def check_message_quality(text, field='message'):
    """Heuristic quality gate for chat messages.

    Rejects three common forms of junk input:
      1. A continuous letter run longer than MAX_ALPHA_RUN (keyboard
         mashing like 'idneibviebvibefvibevebvievowiv...').
      2. A continuous digit run longer than MAX_DIGIT_RUN (phone
         numbers / account numbers like '4384384934573795735384').
      3. Letter ratio under MIN_LETTER_RATIO (mostly punctuation /
         numbers, no question to ask).

    Carefully tuned to PASS legitimate input including emails
    ('info@university.edu'), URLs, prices ('$3,000') and years
    ('2024-2025'). Run unit-test suite if you change the thresholds.
    """
    # 1. Gibberish: one impossibly-long unbroken letter run.
    m = _ALPHA_RUN_RE.search(text)
    if m:
        raise ValidationError(
            field,
            "that looks like gibberish - please ask a real question "
            "(e.g. 'what courses do you offer?')"
        )

    # 2. Phone-number-ish digit string.
    m = _DIGIT_RUN_RE.search(text)
    if m:
        raise ValidationError(
            field,
            'long number sequences (phone numbers, account numbers) '
            "aren't valid questions - try asking in words"
        )

    # 3. Almost-no-letters input (e.g. "########", "..!!.!").
    #    Only enforce on messages with at least 4 chars; very short
    #    inputs like 'ok', '?' are fine even with low letter ratio.
    #    Phone-number-shaped input (digits + space/+/-/parens) is
    #    exempt so users can submit a contact number on its own.
    if _PHONE_LIKE_RE.match(text):
        return
    if len(text) >= 4:
        letters = len(_LETTER_RE.findall(text))
        if letters / max(len(text), 1) < MIN_LETTER_RATIO:
            raise ValidationError(
                field,
                'please ask a question in words - that input is mostly '
                'symbols or numbers'
            )


def parse_bool(value, field):
    """Strict boolean parser. Accepts native bool, the integers 0/1, and
    the case-insensitive strings 'true'/'false'/'1'/'0'/'yes'/'no'.
    Anything else is a ValidationError - we do NOT use Python's truthy
    coercion, because bool('false') returns True, which is a footgun."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ('true', '1', 'yes'):
            return True
        if v in ('false', '0', 'no'):
            return False
    raise ValidationError(field, 'must be a boolean (true/false)')


def parse_float_in_range(value, field, lo=0.0, hi=1.0, allow_none=True):
    """Validate a numeric field that should land within [lo, hi]."""
    if value is None:
        if allow_none:
            return None
        raise ValidationError(field, 'is required')
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValidationError(field, 'must be a number')
    if f < lo or f > hi:
        raise ValidationError(field, f'must be between {lo} and {hi}')
    return f


def validate_intent(value, field, allowed):
    """Ensure value is one of the configured intent tags."""
    if value is None or not isinstance(value, str):
        raise ValidationError(field, 'is required')
    v = value.strip()
    if v not in allowed:
        raise ValidationError(
            field, f"must be one of: {', '.join(sorted(allowed))}"
        )
    return v
