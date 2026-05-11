import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Only download if we haven't already
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# ── Custom stopword list for sentiment analysis ──
# Standard NLTK stopwords MINUS words critical for sentiment meaning.
# Negations (not, no, nor, never, etc.) and intensifiers (very, really, etc.)
# must be PRESERVED because they flip or amplify sentiment.
_KEEP_WORDS = {
    # Negations — these flip sentiment entirely
    'not', 'no', 'nor', 'never', 'neither', 'nobody', 'nothing', 'nowhere',
    'don', 'doesn', 'didn', 'won', 'wouldn', 'couldn', 'shouldn',
    'isn', 'aren', 'wasn', 'weren', 'hasn', 'haven', 'hadn',
    't',  # covers contractions like "don't" → "don" + "t"
    # Intensifiers — these amplify sentiment
    'very', 'really', 'too', 'most', 'more', 'so',
    # Sentiment-carrying words often in stopword lists
    'just', 'only', 'but', 'however', 'against', 'few',
    'above', 'below', 'between',
}

stop_words = set(stopwords.words('english')) - _KEEP_WORDS
stemmer = PorterStemmer()


def clean_text(text):
    text = text.lower()

    # Expand common contractions BEFORE removing punctuation
    contractions = {
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "won't": "will not", "wouldn't": "would not", "couldn't": "could not",
        "shouldn't": "should not", "isn't": "is not", "aren't": "are not",
        "wasn't": "was not", "weren't": "were not", "hasn't": "has not",
        "haven't": "have not", "hadn't": "had not", "can't": "can not",
        "it's": "it is", "i'm": "i am", "i've": "i have",
    }
    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]

    return " ".join(words)
