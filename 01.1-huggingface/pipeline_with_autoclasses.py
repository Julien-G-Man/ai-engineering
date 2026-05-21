from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

my_pipeline = pipeline(
    task="sentiment-analysis",
    model=model,
    tokenizer=tokenizer
)

output = my_pipeline("This course is pretty good, I guess.")
label = output[0]['label']
print(f"Sentiment using AutoClasses: {output}\nLabel: {label}")