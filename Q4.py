import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

positive_reviews = [
    "I absolutely loved this movie! The acting was superb.",
    "A masterpiece! The storyline was captivating.",
    "Brilliant direction and stunning visuals. Highly recommend!",
    "An emotional rollercoaster. Truly unforgettable.",
    "The characters were so relatable. A must-watch!",
    "Fantastic movie! I was hooked from start to finish.",
    "The cinematography was breathtaking. Loved every moment.",
    "A heartwarming story that left me in tears.",
    "The plot twists were mind-blowing. Amazing experience!",
    "One of the best movies I've seen in years.",
    "The soundtrack was incredible. Perfectly matched the scenes.",
    "A beautiful film with a powerful message.",
    "The performances were outstanding. Truly inspiring.",
    "A feel-good movie that I would watch again and again.",
    "The humor was spot on. I laughed so much!",
    "An epic adventure that kept me on the edge of my seat.",
    "The visuals were stunning. A cinematic masterpiece.",
    "The dialogue was so well-written. Very engaging.",
    "A movie that exceeded all my expectations.",
    "The chemistry between the leads was amazing.",
    "A perfect blend of drama and action. Loved it!",
    "The storytelling was so immersive. Couldn't look away.",
    "A movie that touched my heart deeply.",
    "The pacing was perfect. Not a dull moment.",
    "A truly magical experience. Highly recommend!",
    "The acting was phenomenal. A joy to watch!",
    "The movie had a unique and refreshing plot.",
    "The direction was flawless. A cinematic gem.",
    "The visuals were mesmerizing. Truly artistic.",
    "The movie was packed with thrilling moments.",
    "The characters were well-developed and engaging.",
    "The emotional depth of the story was remarkable.",
    "The humor was clever and well-timed.",
    "The movie had a perfect balance of drama and comedy.",
    "The soundtrack added so much to the experience.",
    "The performances were award-worthy.",
    "The movie was a visual treat from start to finish.",
    "The chemistry between the cast was fantastic.",
    "The movie was inspiring and uplifting.",
    "The twists and turns kept me guessing throughout.",
    "The storytelling was masterful and gripping.",
    "The movie had a powerful and meaningful message.",
    "The pacing was spot-on. Never a dull moment.",
    "The cinematography was stunning and innovative.",
    "The movie was a true work of art.",
    "The characters felt so real and relatable.",
    "The movie was an unforgettable experience.",
    "The direction brought the story to life beautifully.",
    "The movie was a perfect blend of heart and humor.",
    "The visuals were breathtaking and immersive.",
]

negative_reviews = [
    "This movie was a complete waste of time.",
    "The acting was so bad, I couldn't take it seriously.",
    "The storyline was boring and predictable.",
    "I regret watching this. It was so disappointing.",
    "The characters were flat and uninteresting.",
    "The visuals were mediocre at best. Not impressive.",
    "The plot made no sense. Very poorly written.",
    "The humor felt forced and awkward.",
    "The pacing was terrible. It dragged on forever.",
    "The ending was so unsatisfying. Left me frustrated.",
    "The soundtrack was annoying and out of place.",
    "The dialogue was cringeworthy and unnatural.",
    "The movie lacked any emotional depth.",
    "The direction was amateurish. Very disappointing.",
    "The special effects were laughably bad.",
    "The performances were wooden and uninspired.",
    "The movie felt like a cheap cash grab.",
    "The twists were predictable and unoriginal.",
    "The editing was sloppy and distracting.",
    "The chemistry between the leads was nonexistent.",
    "The action scenes were poorly choreographed.",
    "The movie was overly long and tedious.",
    "The storytelling was disjointed and confusing.",
    "The movie failed to deliver on its premise.",
    "The overall experience was just plain bad.",
    "The acting was unconvincing and awkward.",
    "The movie lacked any sense of direction.",
    "The visuals were dull and uninspired.",
    "The plot was riddled with inconsistencies.",
    "The humor was childish and unfunny.",
    "The pacing was painfully slow.",
    "The ending was abrupt and unsatisfying.",
    "The soundtrack was distracting and poorly chosen.",
    "The dialogue was cheesy and unrealistic.",
    "The movie felt like a waste of potential.",
    "The characters were one-dimensional and boring.",
    "The special effects were outdated and unimpressive.",
    "The performances were forgettable and bland.",
    "The movie was a chore to sit through.",
    "The twists were laughably bad and predictable.",
    "The editing was chaotic and poorly executed.",
    "The chemistry between the cast was nonexistent.",
    "The action scenes were dull and uninspired.",
    "The movie dragged on far too long.",
    "The storytelling was incoherent and messy.",
    "The movie failed to evoke any emotion.",
    "The direction was uninspired and lackluster.",
    "The visuals were bland and unappealing.",
    "The movie was a disappointment from start to finish.",
    "A truly frustrating and disappointing experience.",
]
reviews = positive_reviews + negative_reviews
sentiments = ["positive"] * 50 + ["negative"] * 50
data = list(zip(reviews, sentiments))
random.shuffle(data)
df = pd.DataFrame(data, columns=["Review", "Sentiment"])
vectorizer = CountVectorizer(stop_words="english", max_features=500)
X = vectorizer.fit_transform(df["Review"])
sentiments = df["Sentiment"]
encoder = LabelEncoder()
encoded_sentiments = encoder.fit_transform(sentiments)
X_train, X_test, y_train, y_test = train_test_split(
    X, encoded_sentiments, test_size=0.2, random_state=42,stratify=encoded_sentiments
)
model = MultinomialNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
def predict_review_sentiment(model, vectorizer, review):
    data = vectorizer.transform([review])
    prediction=model.predict(data)
    return prediction[0]
sample_review = "The movie was absolutely terrible and boring."
print(f"Predicted Sentiment: {predict_review_sentiment(model, vectorizer, sample_review)}")