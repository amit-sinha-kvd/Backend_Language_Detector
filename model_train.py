import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# 1. Load or create a dataset for this project
# For demonstration, we use a small synthetic dataset of common phrases
data = [
    ("Hello, how are you?", "English"),
    ("I love programming", "English"),
    ("The weather is nice today", "English"),
    ("Bonjour, comment ça va ?", "French"),
    ("J'aime la programmation", "French"),
    ("Il fait beau aujourd'hui", "French"),
    ("Hola, ¿cómo estás?", "Spanish"),
    ("Me encanta la programación", "Spanish"),
    ("El clima es agradable hoy", "Spanish"),
    ("Ciao, come stai?", "Italian"),
    ("Amo la programmazione", "Italian"),
    ("Il tempo è bello oggi", "Italian"),
    ("Hallo, wie geht es dir?", "German"),
    ("Ich liebe das Programmieren", "German"),
    ("Das Wetter ist heute schön", "German")
]

X, y = zip(*data)

# 2. Train a scikit-learn model
# We use a Pipeline that combines TF-IDF Vectorization with a Naive Bayes classifier
print("Training language detection model...")
model = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 3), analyzer='char')),
    ('clf', MultinomialNB())
])

model.fit(X, y)

# 3. Save the trained model as model.pkl using joblib
joblib.dump(model, 'model.pkl')
print("Model saved to model.pkl")

# 4. Prints accuracy at the end
# Since it's a small dataset, we'll check performance on the training data
accuracy = model.score(X, y)
print(f"Training Accuracy: {accuracy * 100:.2f}%")
