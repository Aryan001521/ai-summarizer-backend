from transformers import pipeline
from collections import Counter
import re

# Lazy loading
summarizer_pipeline = None


def get_pipeline():
    global summarizer_pipeline

    if summarizer_pipeline is None:
        summarizer_pipeline = pipeline(
            task="summarization",
            model="sshleifer/distilbart-cnn-12-6"
        )

    return summarizer_pipeline


def extract_keywords(text: str, top_n: int = 8) -> list[str]:

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "and", "or", "but", "not", "this", "that", "it", "its",
        "as", "so", "if", "up", "out", "about", "into", "then",
        "than", "also", "i", "we", "you", "he", "she", "they"
    }

    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

    filtered = [w for w in words if w not in stop_words]

    most_common = Counter(filtered).most_common(top_n)

    return [word for word, _ in most_common]


def generate_summary(text: str) -> tuple[str, list[str]]:

    # Short text
    if len(text.split()) < 30:
        keywords = extract_keywords(text)
        return text.strip(), keywords

    summarizer = get_pipeline()

    truncated = text[:2000]

    result = summarizer(
        truncated,
        max_length=100,
        min_length=30,
        do_sample=False
    )

    summary = result[0]["summary_text"]

    keywords = extract_keywords(text)

    return summary, keywords
