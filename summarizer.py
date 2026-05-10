from transformers import pipeline

# Lightweight model
summarizer_pipeline = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)

def generate_summary(text: str):

    if len(text.split()) < 30:
        return text, []

    try:

        text = text[:1500]

        result = summarizer_pipeline(
            text,
            max_length=80,
            min_length=20,
            do_sample=False
        )

        summary = result[0]["summary_text"]

        return summary, []

    except Exception as e:
        print(e)

        return "Error generating summary", []
