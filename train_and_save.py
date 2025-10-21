# train_and_save.py
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from pathlib import Path

# 如果你手头没有数据，我们就临时造一点示例数据
df = pd.DataFrame({
    "text": [
        "Breaking!!! You won't believe this shocking secret!!!",
        "Government releases official economic report for Q3.",
        "Click here to win a free iPhone, limited time!!!",
        "WHO publishes updated health guidelines.",
        "Win $$$ now!!! unbelievable deal click click!!",
        "UN announces new climate initiative for 2030."
    ],
    "label": [1, 0, 1, 0, 1, 0]  # 1=假新闻, 0=真新闻（示例）
})

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.33, random_state=42, stratify=df["label"]
)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1,2), max_features=20000)),
    ("clf", LogisticRegression(max_iter=500))
])

pipe.fit(X_train, y_train)
print("Validation accuracy:", pipe.score(X_test, y_test))

Path("models").mkdir(exist_ok=True)
joblib.dump(pipe, "models/fake_news_pipeline.joblib")
print("Saved → models/fake_news_pipeline.joblib")
