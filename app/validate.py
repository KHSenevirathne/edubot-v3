"""
validate.py - Input validation helpers for EduBot v3

The chat UI is publicly reachable once the bot is deployed, so every
field that crosses the network boundary needs to be sanity-checked
before it touches the model or the database.

Each helper raises ValidationError on failure with a human-readable
message. app.py turns those into HTTP 400 responses with a JSON body.
"""

import re


# Maximum length for any single text field. Picked to be comfortably
# above any genuine question (longest real-world chat message in our
# corpus is ~120 chars) but well below the 1 MB JSON body limit Flask
# would otherwise allow.
MAX_MESSAGE_LEN = 500
MAX_PATTERN_LEN = 200
MIN_TEXT_LEN = 1

# Strip ASCII control codes (NUL, BEL, VT, FF, etc.) but keep newlines
# and tabs because some bot responses include them. Also strip the
# zero-width / direction-override / line-separator / BOM Unicode chars
# used in homograph attacks. We use \uXXXX escapes (not literal Unicode
# characters) to keep this regex robust against editor / encoding
# accidents - U+2028 and U+2029 in particular can break source files
# if dropped in literally.
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


def clean_text_field(value, field, min_len=MIN_TEXT_LEN, max_len=MAX_MESSAGE_LEN):
    """Validate + normalise a free-text field.

    Steps:
      1. Reject None / non-string types.
      2. Strip leading and trailing whitespace.
      3. Drop control characters and zero-width chars.
      4. Enforce min/max length.
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
    return cleaned


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
