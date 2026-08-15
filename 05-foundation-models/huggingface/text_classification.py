from transformers import pipeline

# sentiment analysis
sentiment_analyzer = pipeline(
    task="text-classification",
    model="distilbert-base-uncase-finetuned-sst-2-english"
)

query = "Wi-Fi is slower than a snail today"
print(sentiment_analyzer(query))


# Grammatical correctness
grammer_checker = pipeline(
    task="text-classification",
    model="abdulmatinomotoso/English_Grammer_Checker"
)
print(grammer_checker("He eat pizza every day."))


# Question natural language inference
classifier = pipeline(
    task="text-classification",
    model="cross-encoder/qnli-electra-base"
)
classifier("Where is Seattle located?, Seattle is located in Washington state.")


# dynamic cateory assignment
dyna_classifier = pipeline(
    task="zero-shot-classification",
    model="facebook/bart-large-mnli"
)
text = "hey DataCamp, we would like to feature your courses in our newsletters!"
categories = ["marketing", "sales", "support"]
output = dyna_classifier(text, categories)
print(f"Top Label: {output.get("labels")[0]} with score : {output.get("scores")[0]}")