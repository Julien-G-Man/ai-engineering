"""Summarization reduces a document to a shorter form that preserves key points.
Extractive: selects important sentences or phrases from the original text.
Abstractive: rewrites the content in new words, producing a concise paraphrase.
Set `text` and run to print both summaries.
"""

from transformers import pipeline

extractive_summarizer = pipeline(
    task="summarization",
    model="nyamuda/extractive-summarization"
)

text = "Here is a very large text about Data Science"
summary_text = extractive_summarizer(text)
print(summary_text[0]['summary_text'])


abstractive_summarizer = pipeline(
    task="summarization",
    model="sshleifer/distilbart-cnn-12-6",
    min_new_tokens=1,
    max_new_tokens=10
)
summary = abstractive_summarizer(text)
print(summary[0]['summary_text'])