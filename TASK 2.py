import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\saite\Desktop\reviews.csv", encoding='latin1')

# Show sample and columns
print("Sample data:")
print(df.head())
print("\nColumns:", list(df.columns))

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Check class distribution (handle possible column name issues)
sentiment_col = None
review_col = None
for col in df.columns:
    if col.strip().lower() == 'sentiment':
        sentiment_col = col
    if col.strip().lower() == 'review':
        review_col = col
if sentiment_col is None or review_col is None:
    raise KeyError("Could not find 'Review' or 'Sentiment' columns. Found columns: {}".format(list(df.columns)))

print("\nClass distribution:\n", df[sentiment_col].value_counts())

# Drop rows with missing reviews or sentiments
df = df.dropna(subset=[review_col, sentiment_col])

# Convert reviews to lowercase
df[review_col] = df[review_col].str.lower()

# Encode labels (e.g., positive → 1, negative → 0)
le = LabelEncoder()
df[sentiment_col] = le.fit_transform(df[sentiment_col])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df[review_col], df[sentiment_col], test_size=0.2, random_state=42, stratify=df[sentiment_col])

# TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_df=0.95)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# Predict
y_pred = model.predict(X_test_tfidf)

# Evaluate
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
