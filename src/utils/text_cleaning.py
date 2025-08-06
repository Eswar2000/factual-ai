import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text_aggressive(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text) # remove non-alphabetic
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words
    ]
    return " ".join(tokens)

def preprocess_text_light(text: str) -> str:
    text = str(text).strip()
    return re.sub(r"\s+", " ", text) # collapse multiple spaces

