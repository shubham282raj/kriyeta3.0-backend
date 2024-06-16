import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def sentiment_analysis(text):
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    positive_label = 'positive'
    neutral_label = 'negative'
    negative_label = 'negative'

    positive_index = [i for i, label in config.id2label.items() if label == positive_label]
    neutral_index = [i for i, label in config.id2label.items() if label == neutral_label]
    negative_index = [i for i, label in config.id2label.items() if label == negative_label]

    positive_score = scores[positive_index]*10
    neutral_score = scores[neutral_index]*10
    negative_score = scores[negative_index]*10
    
    return {"positive": str(float(positive_score)), "neutral": str(float(neutral_score)), "negative": str(float(negative_score))}






