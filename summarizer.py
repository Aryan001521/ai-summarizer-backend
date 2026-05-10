from transformers import pipeline
from collections import Counter
import re
summarizer_pipeline = pipeline(
    task="summarization",
    model="facebook/bart-large-cnn"
)
 
def extract_keywords(text: str, top_n: int = 8) -> list[str]:
    # Stop words jo skip karne hain
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
    # Text bahut chota hai toh summarize mat karo
    if len(text.split()) < 30:
        keywords = extract_keywords(text)
        return text.strip(), keywords
 
    # BART ka max input ~1024 tokens hai, isliye truncate karo
    truncated = text[:3000]   
 
    result = summarizer_pipeline(
        truncated,
        max_length=150,
        min_length=40,
        do_sample=False
    )
 
    summary = result[0]["summary_text"]
    keywords = extract_keywords(text)
 
    return summary, keywords