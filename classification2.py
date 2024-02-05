
import pandas as pd
import numpy as np
from ast import literal_eval

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from utils.embeddings_utils import plot_multiclass_precision_recall

datafile_path = "fine_food_reviews_with_embeddings_1k.csv"

df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)

# Convert target labels to integers
label_encoder = LabelEncoder()
df['Score'] = label_encoder.fit_transform(df['Score'])

# Convert the list of arrays to a 2D array
X = np.vstack(df.embedding.values)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, df.Score, test_size=0.2, random_state=42
)

# train random forest classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
probas = clf.predict_proba(X_test)

report = classification_report(y_test, preds, zero_division=1)
print(report)

# Assuming plot_multiclass_precision_recall is correctly implemented
#plot_multiclass_precision_recall(probas, y_test, label_encoder.classes_, clf)
plot_multiclass_precision_recall(probas, y_test, [1, 2, 3], clf)
