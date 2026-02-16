import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

try:
    df = pd.read_csv(
        'spam.csv',
        encoding='latin-1',
        sep='\t',
        header=None, 
        names=['label', 'message'] 
    )
except FileNotFoundError:
    print("Error: 'spam.csv' not found.")
    print("Please download the 'UCI SMS Spam Collection' dataset,")
    print("rename it to 'spam.csv', and place it in the same folder.")
    exit()

print("Data loaded successfully. First 5 rows:")
print(df.head())

# convert label into numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

#splitting and training
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#ml model
pipe_nb = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True)),
    ('model', MultinomialNB())
])

pipe_log_reg = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True)),
    ('model', LogisticRegression(solver='liblinear'))
])

pipe_svm = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True)),
    ('model', SVC(kernel='linear'))
])



print("\n--- Training and Evaluating Models ---")

pipe_nb.fit(X_train, y_train)
preds_nb = pipe_nb.predict(X_test)
print("\n--- Naive Bayes Results ---")
print(classification_report(y_test, preds_nb))

pipe_log_reg.fit(X_train, y_train)
preds_log_reg = pipe_log_reg.predict(X_test)
print("--- Logistic Regression Results ---")
print(classification_report(y_test, preds_log_reg)) 

pipe_svm.fit(X_train, y_train)
preds_svm = pipe_svm.predict(X_test)
print("--- SVM Results ---")
print(classification_report(y_test, preds_svm))


print("\n--- Testing on New Emails ---")

model = pipe_svm 

new_emails = [
    """Your Upstox account is inactive because you haven't traded on it for over 24 months i.e. from 05 July 2023. This is a mandate from the Exchange to ensure your account remains secure.


To reactivate it please redo the KYC process by following these steps:""",
]

predictions = model.predict(new_emails)

for email, prediction in zip(new_emails, predictions):
    result = 'Spam' if prediction == 1 else 'Ham'
    print(f"Email: '{email}'  ->  Prediction: {result}")