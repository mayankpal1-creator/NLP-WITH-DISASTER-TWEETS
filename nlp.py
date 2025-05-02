import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Fill missing values
train_data['keyword'].fillna('none', inplace=True)
train_data['location'].fillna('unknown', inplace=True)
test_data['keyword'].fillna('none', inplace=True)
test_data['location'].fillna('unknown', inplace=True)

# Combine text fields
train_data['combined_text'] = train_data['text'] + ' ' + train_data['keyword'] + ' ' + train_data['location']
test_data['combined_text'] = test_data['text'] + ' ' + test_data['keyword'] + ' ' + test_data['location']

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_data['combined_text'])
X_test = vectorizer.transform(test_data['combined_text'])

# Target labels
y_train = train_data['target']

# Split the training data for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_split, y_train_split)

# Predict on validation set
y_val_pred = model.predict(X_val)

# Evaluate the model
print("Validation Set Evaluation")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
print("Classification Report:\n", classification_report(y_val, y_val_pred))

# Predict on test set
test_pred = model.predict(X_test)

# Map predictions to 0 and 1
test_pred_labels = np.where(test_pred == 1, 'YES', 'NO')

# Create a submission file with the full tweet text and prediction
submission = pd.DataFrame({
    'id': test_data['id'],
    'text': test_data['text'],
    'target': test_pred_labels
})
submission.to_csv('submission.csv', index=False)

print("Submission file created: submission.csv")
