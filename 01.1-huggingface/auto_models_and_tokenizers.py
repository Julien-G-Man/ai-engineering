"""Shows Hugging Face Auto* classes: AutoModel, AutoTokenizer, AutoConfig.
Auto classes automatically select the correct architecture/tokenizer
from a model identifier, avoiding hardcoded class names.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# download a pre-trained text classification model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

# Retrieve the tokenizer paired with the model
model = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)


tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
tokens = tokenizer.tokenize("AI: Helping robots think and humans overthink :)")
print(tokens)