import re
from pathlib import Path

import pandas as pd

SHOW_PLOTS = True
SAVE_PLOTS = True
RUN_TF_MODELS = False
RUN_BERT_MODEL = False
RUN_RANDOM_FOREST = False

def _safe_import_nltk():
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
    except Exception as exc:
        print("NLTK not available; continuing without stopwords/lemmatizer.", exc)
        return None, None, None

    stopwords_ok = True
    wordnet_ok = True
    try:
        nltk.data.find("corpora/stopwords")
    except Exception:
        stopwords_ok = False
    try:
        nltk.data.find("corpora/wordnet")
    except Exception:
        wordnet_ok = False

    return (
        nltk,
        stopwords if stopwords_ok else None,
        WordNetLemmatizer if wordnet_ok else None,
    )


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def preprocess_text(text: str, stop_words: set, lemmatizer) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation/numbers
    words = text.split()
    if lemmatizer is not None:
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    else:
        words = [w for w in words if w not in stop_words]
    return ' '.join(words)


def textcleaning(text: str) -> str:
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

DATA_DIR = Path(__file__).resolve().parent
fake_path = DATA_DIR / "Fake.csv.zip"
true_path = DATA_DIR / "True.csv.zip"

# Load datasets
fake_df = _load_dataset(fake_path)
true_df = _load_dataset(true_path)

# Add labels
fake_df['label'] = 1
true_df['label'] = 0

# Combine and shuffle
data = pd.concat([fake_df, true_df], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

required_cols = {"title", "text"}
missing = required_cols - set(data.columns)
if missing:
    raise KeyError(f"Missing required columns: {missing}")

data['title'] = data['title'].fillna("").astype(str)
data['text'] = data['text'].fillna("").astype(str)

# Dataset summary
def print_dataset_summary(name: str, df: pd.DataFrame) -> None:
    print(f"=== {name} ===")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Dtypes:", df.dtypes.to_dict())
    print("Missing per column:", df.isna().sum().to_dict())
    print("Total missing:", int(df.isna().sum().sum()))
    print("")


print_dataset_summary("Fake", fake_df)
print_dataset_summary("True", true_df)
print_dataset_summary("Combined", data)

# Data visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use("seaborn-v0_8")

    plots_dir = DATA_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    def _finalize_plot(filename: str) -> None:
        if SAVE_PLOTS:
            plt.savefig(plots_dir / filename, dpi=150)
        if SHOW_PLOTS:
            plt.show()
        plt.close()

    # Class distribution
    plt.figure(figsize=(6, 4))
    class_counts = data['label'].value_counts().sort_index()
    sns.barplot(x=class_counts.index.map({0: "True", 1: "Fake"}), y=class_counts.values)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    _finalize_plot("class_distribution.png")

    # Subject distribution (top 10)
    if "subject" in data.columns:
        plt.figure(figsize=(8, 4))
        subject_counts = data["subject"].value_counts().head(10)
        sns.barplot(x=subject_counts.values, y=subject_counts.index)
        plt.title("Top 10 Subjects")
        plt.xlabel("Count")
        plt.ylabel("Subject")
        plt.tight_layout()
        _finalize_plot("top_subjects.png")

    # Text length distribution
    plt.figure(figsize=(8, 4))
    text_lengths = data["text"].astype(str).str.len()
    sns.histplot(text_lengths, bins=50, kde=True)
    plt.title("Text Length Distribution")
    plt.xlabel("Text Length (chars)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    _finalize_plot("text_length_distribution.png")

    # Text length by class (box plot)
    plt.figure(figsize=(6, 4))
    data["text_len"] = data["text"].astype(str).str.len()
    sns.boxplot(x=data["label"].map({0: "True", 1: "Fake"}), y=data["text_len"])
    plt.title("Text Length by Class")
    plt.xlabel("Class")
    plt.ylabel("Text Length (chars)")
    plt.tight_layout()
    _finalize_plot("text_length_by_class.png")

    # Word count distribution
    plt.figure(figsize=(8, 4))
    data["word_count"] = data["text"].astype(str).str.split().str.len()
    sns.histplot(data["word_count"], bins=50, kde=True)
    plt.title("Word Count Distribution")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    _finalize_plot("word_count_distribution.png")

    # Top 20 words (overall)
    try:
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(max_features=20, stop_words="english")
        word_counts = vectorizer.fit_transform(data["processed"])
        words = vectorizer.get_feature_names_out()
        counts = word_counts.toarray().sum(axis=0)
        word_freq = pd.Series(counts, index=words).sort_values(ascending=True)

        plt.figure(figsize=(8, 6))
        sns.barplot(x=word_freq.values, y=word_freq.index)
        plt.title("Top 20 Words (Processed Text)")
        plt.xlabel("Count")
        plt.ylabel("Word")
        plt.tight_layout()
        _finalize_plot("top_20_words.png")
    except Exception as exc:
        print("Word frequency plot skipped.", exc)
except Exception as exc:
    print("Visualization skipped (matplotlib/seaborn not available).", exc)

# Preprocessing function
_nltk, stopwords, WordNetLemmatizer = _safe_import_nltk()
stop_words = set(stopwords.words('english')) if stopwords else set()
lemmatizer = WordNetLemmatizer() if WordNetLemmatizer else None

# Apply preprocessing to title + text
data['content'] = data['title'] + " " + data['text']
data['processed'] = data['content'].apply(
    lambda s: preprocess_text(textcleaning(s), stop_words, lemmatizer)
)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(data['processed'])
y = data['label'].values

from sklearn.model_selection import train_test_split, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

models = [
    ("Logistic Regression", LogisticRegression(max_iter=1000), False),
    ("Linear SVM", LinearSVC(dual="auto"), False),
    ("Naive Bayes", MultinomialNB(), False),
]
if RUN_RANDOM_FOREST:
    models.append(
        ("Random Forest", RandomForestClassifier(n_estimators=100, n_jobs=1), True)
    )

for name, model, needs_dense in models:
    Xtr = X_train.toarray() if needs_dense else X_train
    Xte = X_test.toarray() if needs_dense else X_test
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    print(f"=== {name} ===")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred), "\n")

    if name == "Logistic Regression":
        try:
            y_score = model.predict_proba(Xte)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, color="blue", label=f"ROC AUC = {roc_auc:.4f}")
            plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
            plt.title("ROC Curve: Logistic Regression Performance")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            plt.tight_layout()
            _finalize_plot("roc_curve_logistic_regression.png")
        except Exception as exc:
            print("ROC curve skipped.", exc)

# Grid search (sparse-friendly models)
def run_grid_search(model_name: str, estimator, param_grid: dict) -> None:
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="f1",
        n_jobs=1,
        cv=5
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    print("=== HYPERPARAMETER TUNING RESULTS (GridSearchCV) ===")
    results_df = pd.DataFrame(grid.cv_results_)
    if "param_C" in results_df.columns:
        summary = (
            results_df.groupby("param_C")[["mean_test_score", "std_test_score"]]
            .mean()
            .reset_index()
            .sort_values("param_C")
        )
        print("C Value  Mean F1-Score (5-Fold CV)  Std Dev F1-Score")
        for _, row in summary.iterrows():
            print(
                f"{row['param_C']:>6}     "
                f"{row['mean_test_score']:.6f}             "
                f"{row['std_test_score']:.6f}"
            )
    else:
        top_cols = ["mean_test_score", "std_test_score", "params"]
        top_results = results_df.sort_values(
            "mean_test_score", ascending=False
        )[top_cols].head(5)
        print("Top 5 CV results:\n", top_results.to_string(index=False))

    print("\n--- Best Hyperparameters found by GridSearchCV ---")
    print(grid.best_params_)
    print(f"Best F1-score during tuning: {grid.best_score_:.4f}")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred), "\n")


run_grid_search(
    "Logistic Regression",
    LogisticRegression(max_iter=2000, solver="liblinear"),
    {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "class_weight": [None, "balanced"],
        "penalty": ["l1", "l2"],
    }
)
run_grid_search(
    "Linear SVM",
    LinearSVC(dual="auto"),
    {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "class_weight": [None, "balanced"],
    }
)

try:
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    _tf_available = True
except Exception as exc:
    print("TensorFlow not available; skipping LSTM model.", exc)
    _tf_available = False

if _tf_available and RUN_TF_MODELS:
    # Tokenization
    max_words = 10000
    max_len = 200

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data['processed'])
    sequences = tokenizer.texts_to_sequences(data['processed'])
    X_seq = pad_sequences(sequences, maxlen=max_len)

    # Split
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
        X_seq, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model architecture
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    model.summary()

    # Training
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    # Evaluation
    loss, acc = model.evaluate(X_test_seq, y_test_seq, verbose=0)
    print(f"LSTM Test Accuracy: {acc:.4f}")


try:
    import tensorflow as tf
    from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
    _bert_available = True
except Exception as exc:
    print("Transformers/TensorFlow not available; skipping DistilBERT.", exc)
    _bert_available = False

if _bert_available and RUN_BERT_MODEL:
    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = TFDistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )

    # Split indices for train/test
    train_idx, test_idx = train_test_split(
        range(len(data)), test_size=0.2, random_state=42, stratify=y
    )

    # Tokenize sequences
    train_encodings = tokenizer(
        data.iloc[train_idx]['processed'].tolist(),
        truncation=True,
        padding=True,
        max_length=512
    )
    test_encodings = tokenizer(
        data.iloc[test_idx]['processed'].tolist(),
        truncation=True,
        padding=True,
        max_length=512
    )

    # Convert to TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        y[train_idx]
    )).shuffle(1000).batch(16)

    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
        y[test_idx]
    )).batch(16)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Train
    model.fit(train_dataset, epochs=3, validation_data=test_dataset)

    # Evaluate
    results = model.evaluate(test_dataset)
    print(f"DistilBERT Test Accuracy: {results[1]:.4f}")

