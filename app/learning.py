import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import database as db   # noqa: E402

# After this many user-taught patterns, retrain automatically.
# Lowering this to 1 makes the loop visible during a viva demo.
AUTO_RETRAIN_THRESHOLD = 5


def record_feedback(user_message, bot_response, predicted_intent,
                    confidence, helpful, expected_intent=None):
    """Persist a thumbs-up / thumbs-down (Tier 1).

    A thumbs-down with an expected_intent becomes a PENDING REVIEW row
    in learned_patterns (approved=0). It does NOT enter training until
    an admin approves it.

    Returns dict with feedback_id, retrained (always False here),
    pending_review count and pending_patterns count.
    """
    feedback_id = db.log_feedback(
        user_message=user_message,
        bot_response=bot_response,
        predicted_intent=predicted_intent,
        confidence=confidence,
        helpful=helpful,
        expected_intent=expected_intent,
    )

    if not helpful and expected_intent:
        db.add_learned_pattern(
            pattern=user_message,
            intent=expected_intent,
            source='feedback_correction',
            approved=False,           # waits for admin review
        )

    return {
        'feedback_id': feedback_id,
        'retrained': False,
        'pending_review': db.count_pending_review(),
        'learned_patterns_pending': db.count_pending_patterns(),
    }


def teach(pattern, intent):
    """Admin direct-teach (Tier 2).

    Admins are trusted, so the row is inserted as approved=1 and may
    trigger an immediate auto-retrain.
    """
    pattern_id = db.add_learned_pattern(
        pattern=pattern,
        intent=intent,
        source='direct_teach',
        approved=True,
    )
    retrained = _maybe_auto_retrain()
    return {
        'pattern_id': pattern_id,
        'retrained': retrained,
        'learned_patterns_pending': db.count_pending_patterns(),
    }


def approve_suggestion(pattern_id):
    """Admin approves a pending suggestion (Tier 2).

    Flips approved to 1 and may trigger an auto-retrain if enough
    approved patterns have piled up.
    """
    if not db.approve_pattern(pattern_id):
        return {'approved': False, 'reason': 'not_found'}
    retrained = _maybe_auto_retrain()
    return {
        'approved': True,
        'retrained': retrained,
        'learned_patterns_pending': db.count_pending_patterns(),
        'pending_review': db.count_pending_review(),
    }


def discard_suggestion(pattern_id):
    """Admin discards a pending suggestion. Row is deleted entirely."""
    ok = db.discard_pattern(pattern_id)
    return {
        'discarded': ok,
        'pending_review': db.count_pending_review(),
    }


def manual_retrain():
    """Force a retrain regardless of the threshold. Returns model name."""
    return _run_training()


def _maybe_auto_retrain():
    """Retrain if pending pattern count crossed the threshold."""
    if db.count_pending_patterns() >= AUTO_RETRAIN_THRESHOLD:
        _run_training()
        return True
    return False


def _run_training():
    """Invoke train.py's pipeline. Imported lazily so a malformed model
    file doesn't break Flask startup before the first call."""
    from train import train_and_evaluate
    return train_and_evaluate(verbose=False)


# Sanity check - lets you run `python app/learning.py` to see DB state.
if __name__ == "__main__":
    db.init_schema()
    print("Current DB stats:")
    for k, v in db.stats().items():
        print(f"  {k}: {v}")
