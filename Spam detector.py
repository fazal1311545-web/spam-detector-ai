import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("spam.csv")

print("\n====== SPAM DETECTOR ======\n")
print("📊 Dataset Info:")
print("Total Messages:", len(data))

counts = data['label'].value_counts()
print("Spam Messages:", counts.get('spam', 0))
print("Safe Messages:", counts.get('safe', 0))

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])

y = data['label']

model = MultinomialNB()
model.fit(X, y)

print("\n✅ Model is Ready!")
print("Type 'exit' to quit\n")

while True:
    msg = input("Enter message: ").lower().strip()

    if msg in ["exit"]:
        print("👋 Program ended. Goodbye!")
        break

    if msg == "":
        print("⚠ Please enter a valid message")
        continue

    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)

    print("Result:", prediction[0])