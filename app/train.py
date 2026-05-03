"""
train.py - Model Training Pipeline for EduBot v3

Loads training data from TWO sources:
  1. data/intents.json   - hand-curated patterns (the seed corpus)
  2. learned_patterns DB - patterns the bot has been TAUGHT at runtime

Trains three classifiers (Multinomial Naive Bayes, Linear SVM, Random
Forest), compares them by 5-fold cross-validation accuracy, and pickles
the winner along with its TF-IDF vectoriser.

Running this script is the closing step of EduBot's machine-learning
loop:
    user feedback -> learned_patterns -> retrain -> better model
"""

import json
import pickle
import os
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# Local imports - these have to come AFTER sys.path is set so that
# `python app/train.py` works from any working directory.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import clean_text  # noqa: E402
import database as db              # noqa: E402


def load_intents(filepath):
    """Read the intents.json seed corpus."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_training_data(intents_data, learned_rows):
    """Merge seed patterns and DB-learned patterns into (X, y)."""
    patterns, tags = [], []

    # Source 1: hand-written intents.json
    for intent in intents_data['intents']:
        for pattern in intent['patterns']:
            patterns.append(clean_text(pattern))
            tags.append(intent['tag'])

    # Source 2: rows from learned_patterns table (the ML-loop output).
    # These survive across retrains because they live in the DB.
    for row in learned_rows:
        patterns.append(clean_text(row['pattern']))
        tags.append(row['intent'])

    return patterns, tags


def train_and_evaluate(verbose=True):
    """End-to-end training pipeline. Returns the saved model name."""
    log = print if verbose else (lambda *a, **k: None)

    log("=" * 60)
    log("  EduBot v3 - Model Training Pipeline")
    log("=" * 60)

    # Step 1: Load both data sources
    log("\nStep 1: Loading training data...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    intents_path = os.path.join(base_dir, 'data', 'intents.json')
    intents_data = load_intents(intents_path)

    db.init_schema()
    # Only approved patterns enter training. Pending-review rows from
    # end-user feedback stay out until an admin signs off.
    learned_rows = db.get_learned_patterns(approved_only=True)

    seed_count = sum(len(i['patterns']) for i in intents_data['intents'])
    intent_count = len(intents_data['intents'])
    log(f"   - {intent_count} intents loaded from intents.json")
    log(f"   - {seed_count} seed patterns")
    log(f"   - {len(learned_rows)} approved patterns learned at runtime")

    # Step 2: Preprocess and merge
    log("\nStep 2: Preprocessing patterns...")
    patterns, tags = prepare_training_data(intents_data, learned_rows)
    log(f"   - {len(patterns)} total preprocessed patterns")

    # Step 3: TF-IDF
    log("\nStep 3: Vectorising with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(patterns)
    y = np.array(tags)
    log(f"   - Feature matrix shape: {X.shape}")
    log(f"   - Vocabulary size: {len(vectorizer.vocabulary_)}")

    # Step 4: Train/test split (stratified so every class appears in both)
    log("\nStep 4: Splitting data (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log(f"   - Training samples: {X_train.shape[0]}")
    log(f"   - Testing samples:  {X_test.shape[0]}")

    # Step 5: Train and compare three classifiers
    log("\nStep 5: Training models...")
    models = {
        'Naive Bayes':   MultinomialNB(),
        'SVM':           SVC(kernel='linear', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    }
    results = {}

    for name, model in models.items():
        log(f"\n   Training {name}...")
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        cv_scores = cross_val_score(model, X, y, cv=5)
        results[name] = {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
        }
        log(f"   - Train accuracy:    {train_acc * 100:.1f}%")
        log(f"   - Test accuracy:     {test_acc * 100:.1f}%")
        log(f"   - 5-fold CV accuracy: "
            f"{cv_scores.mean() * 100:.1f}% (+/- {cv_scores.std() * 100:.1f}%)")

    # Step 6: Pick best model by CV accuracy
    log("\n" + "=" * 60)
    log("  MODEL COMPARISON")
    log("=" * 60)
    log(f"\n   {'Model':<20} {'Train':>8} {'Test':>8} {'CV Mean':>8}")
    log(f"   {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 8}")
    best_name, best_cv = None, 0
    for name, res in results.items():
        log(f"   {name:<20} "
            f"{res['train_acc'] * 100:>7.1f}% "
            f"{res['test_acc']  * 100:>7.1f}% "
            f"{res['cv_mean']   * 100:>7.1f}%")
        if res['cv_mean'] > best_cv:
            best_cv = res['cv_mean']
            best_name = name
    log(f"\n   Best Model: {best_name} (CV accuracy: {best_cv * 100:.1f}%)")

    # Step 7: Detailed classification report
    best_model = results[best_name]['model']
    y_pred = best_model.predict(X_test)
    log("\n" + "=" * 60)
    log(f"  CLASSIFICATION REPORT ({best_name})")
    log("=" * 60)
    log(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    log("CONFUSION MATRIX (rows = actual, columns = predicted):")
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    log(f"   {'':>15}" + "".join(f"{lab[:8]:>10}" for lab in labels))
    for i, lab in enumerate(labels):
        log(f"   {lab:>15}" + "".join(f"{cm[i][j]:>10}" for j in range(len(labels))))

    # Step 8: Persist model + vectoriser
    log(f"\nStep 8: Saving best model ({best_name})...")
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, 'chatbot_model.pkl')
    vec_path = os.path.join(models_dir, 'vectorizer.pkl')
    info_path = os.path.join(models_dir, 'model_info.txt')

    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    with open(vec_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(
            f"Model: {best_name}\n"
            f"Train accuracy: {results[best_name]['train_acc'] * 100:.1f}%\n"
            f"Test accuracy:  {results[best_name]['test_acc'] * 100:.1f}%\n"
            f"CV accuracy:    {results[best_name]['cv_mean'] * 100:.1f}%\n"
            f"Intents: {intent_count}\n"
            f"Total patterns: {len(patterns)}\n"
            f"Seed patterns: {seed_count}\n"
            f"Learned patterns: {len(learned_rows)}\n"
        )

    # Step 9: Mark all learned patterns as baked into the model.
    db.mark_patterns_used()

    log(f"   - Model saved to:      {model_path}")
    log(f"   - Vectorizer saved to: {vec_path}")
    log(f"   - Info saved to:       {info_path}")

    log("\n" + "=" * 60)
    log("  Training complete. Run `python app.py` to launch the chatbot.")
    log("=" * 60)

    return best_name


if __name__ == "__main__":
    train_and_evaluate()
