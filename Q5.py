from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import List
good_feedback = [
    "The product quality is excellent!",
    "I am very satisfied with this purchase.",
    "Amazing product! Highly recommend.",
    "Works perfectly as advertised.",
    "Exceeded my expectations. Great value!",
    "The customer service was outstanding.",
    "Very durable and reliable product.",
    "The design is sleek and modern.",
    "Easy to use and very effective.",
    "I would definitely buy this again.",
    "The packaging was very professional.",
    "This product is worth every penny.",
    "The performance is top-notch.",
    "I am extremely happy with this product.",
    "The product arrived on time and in perfect condition.",
    "The instructions were clear and easy to follow.",
    "The build quality is fantastic.",
    "This is the best product I have ever used.",
    "The product is very user-friendly.",
    "I am impressed with the attention to detail.",
    "The product works flawlessly.",
    "The price is very reasonable for the quality.",
    "I love the features of this product.",
    "The product is exactly as described.",
    "The product has made my life so much easier.",
    "The product is very versatile.",
    "The product is lightweight and portable.",
    "The product is very energy-efficient.",
    "The product is very stylish.",
    "The product is very comfortable to use.",
    "The product is very safe and secure.",
    "The product is very innovative.",
    "The product is very affordable.",
    "The product is very reliable.",
    "The product is very convenient.",
    "The product is very effective.",
    "The product is very easy to clean.",
    "The product is very eco-friendly.",
    "The product is very compact.",
    "The product is very quiet.",
    "The product is very fast.",
    "The product is very powerful.",
    "The product is very accurate.",
    "The product is very flexible.",
    "The product is very customizable.",
    "The product is very intuitive.",
    "The product is very responsive.",
    "The product is very efficient.",
    "The product is very robust.",
    "The product is very dependable.",
]

bad_feedback = [
    "The product quality is terrible.",
    "I am very disappointed with this purchase.",
    "Awful product! Do not recommend.",
    "Does not work as advertised.",
    "Failed to meet my expectations. Waste of money!",
    "The customer service was unhelpful.",
    "Very fragile and unreliable product.",
    "The design is outdated and ugly.",
    "Difficult to use and ineffective.",
    "I would never buy this again.",
    "The packaging was very poor.",
    "This product is not worth the price.",
    "The performance is subpar.",
    "I am extremely unhappy with this product.",
    "The product arrived late and damaged.",
    "The instructions were confusing and hard to follow.",
    "The build quality is terrible.",
    "This is the worst product I have ever used.",
    "The product is not user-friendly.",
    "I am disappointed with the lack of attention to detail.",
    "The product does not work properly.",
    "The price is too high for the quality.",
    "I hate the features of this product.",
    "The product is not as described.",
    "The product has made my life more difficult.",
    "The product is very limited in functionality.",
    "The product is heavy and hard to carry.",
    "The product is not energy-efficient.",
    "The product is ugly and unappealing.",
    "The product is uncomfortable to use.",
    "The product is unsafe and insecure.",
    "The product is not innovative.",
    "The product is overpriced.",
    "The product is unreliable.",
    "The product is inconvenient.",
    "The product is ineffective.",
    "The product is hard to clean.",
    "The product is not eco-friendly.",
    "The product is bulky.",
    "The product is noisy.",
    "The product is slow.",
    "The product is weak.",
    "The product is inaccurate.",
    "The product is inflexible.",
    "The product is not customizable.",
    "The product is confusing to use.",
    "The product is unresponsive.",
    "The product is inefficient.",
    "The product is flimsy.",
    "The product is undependable.",
]
feedback = good_feedback + bad_feedback
labels = ["good"] * len(good_feedback) + ["bad"] * len(bad_feedback)
data = list(zip(feedback, labels))
random.shuffle(data)
df = pd.DataFrame(data, columns=["Feedback", "Label"])
print(df)
vectorizer = TfidfVectorizer(stop_words="english", max_features=300)
X = vectorizer.fit_transform(df["Feedback"])
X_train, X_test, y_train, y_test = train_test_split(df["Feedback"], df["Label"],test_size=0.25, random_state=42,stratify=df["Label"])

pipe = Pipeline([
    ("vectorizer", TfidfVectorizer(stop_words="english", max_features=300,lowercase=True)),
    ("classifier",LogisticRegression())
])
pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, predictions, digits=2))
def text_preprocess_vectorize(texts: List[str], vectorizer: TfidfVectorizer) :
    return vectorizer.transform(texts)