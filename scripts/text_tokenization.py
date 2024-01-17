import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Load the spacy small English model 
spacy_nlp = spacy.load('en_core_web_sm')
# Compile the regular expression to use
re_pattern = re.compile(r'[\W_]+')

def tokenizeText(text: str, lemmas: bool) -> str:
    """
    Tokenizes the input text and returns a processed string.
    ... [rest of the docstring] ...
    """
    text = re_pattern.sub(' ', text)
    doc = spacy_nlp(text)
    if lemmas:
        tokens = [token.lemma_.lower() for token in doc if token.lemma_.lower() not in STOP_WORDS]
        return ' '.join(tokens)
    else:
        tokens = [token.text.lower() for token in doc if token.text.lower() not in STOP_WORDS]
        return tokens