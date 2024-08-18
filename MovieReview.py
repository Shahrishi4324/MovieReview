import spacy
from sklearn.datasets import load_files

# Load the SpaCy English model
nlp = spacy.load('en_core_web_sm')

# Load movie review data
movie_data = load_files('./data', shuffle=True)
X, y = movie_data.data, movie_data.target

# Convert text data to lowercase
X = [doc.decode('utf-8').lower() for doc in X]

# Preprocess the text data using SpaCy
def preprocess(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

X_preprocessed = [preprocess(doc) for doc in X]

# Print the first preprocessed review
print(X_preprocessed[0])

from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Transform the preprocessed data into TF-IDF features
X_tfidf = vectorizer.fit_transform(X_preprocessed)

# Print the shape of the TF-IDF matrix
print(f"Shape of TF-IDF matrix: {X_tfidf.shape}")