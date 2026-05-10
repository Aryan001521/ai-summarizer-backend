from transformers import pipeline
from collections import Counter
import re

# ✅ Lightweight summarization model
summarizer_pipeline = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)

# ---------------- KEYWORDS ----------------
def extract_keywords(text: str, top_n: int = 8) -> list[str]:

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with",
        "at", "by", "from", "and", "or", "but",
        "not", "this", "that", "it", "its", "as",
        "so", "if", "up", "out", "about", "into",
        "then", "than", "also", "i", "we", "you",
        "he", "she", "they"
    }

    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

    filtered = [
        word for word in words
        if word not in stop_words
    ]

    most_common = Counter(filtered).most_common(top_n)

    return [word for word, _ in most_common]


# ---------------- SUMMARY ----------------
def generate_summary(text: str):

    # Small text → no summarize
    if len(text.split()) < 30:
        keywords = extract_keywords(text)
        return text.strip(), keywords

    try:

        # Limit text size
        truncated_text = text[:2000]

        result = summarizer_pipeline(
            truncated_text,
            max_length=120,
            min_length=40,
            do_sample=False
        )

        summary = result[0]["summary_text"]

        keywords = extract_keywords(text)

        return summary, keywords

    except Exception as e:
        print("Summarization Error:", e)

        return (
            "Error generating summary",
            []
        )
