import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

def getValueBetweenTexts(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""

def lemmatizeVerb(verb):
    """Lemmatize the given verb and return the base form of the verb"""
    lemmatizer = WordNetLemmatizer()
    base_form = lemmatizer.lemmatize(verb, pos='v')
    return base_form

def getObjectPronun(subjext_pronoun):
    """Get object pronoun of the given subject pronoun"""
    dict = {
        "i": "me",
        "you": "you",
        "he": "him",
        "she": "her",
        "it": "it",
        "we": "us",
        "they": "them"
    }
    if subjext_pronoun in dict.keys():
        return dict[subjext_pronoun]
    else:
        raise ValueError("Invalid pronoun")