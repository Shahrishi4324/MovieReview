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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Print the training accuracy
train_accuracy = model.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate and print the test accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Print classification report
print(classification_report(y_test, y_pred))

# Print confusion matrix
print(confusion_matrix(y_test, y_pred))

# Test the model with new reviews
new_reviews = ["This movie was fantastic! I loved it.", "The movie was terrible, I hated it."]
new_reviews_preprocessed = [preprocess(review) for review in new_reviews]
new_reviews_tfidf = vectorizer.transform(new_reviews_preprocessed)

# Predict sentiment
new_predictions = model.predict(new_reviews_tfidf)

# Output the predictions
for review, sentiment in zip(new_reviews, new_predictions):
    print(f"Review: {review} \nSentiment: {'Positive' if sentiment == 1 else 'Negative'}\n")
