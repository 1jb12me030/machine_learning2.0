import pandas as pd
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, precision_recall_curve
from utils.embeddings_utils import cosine_similarity, get_embedding

EMBEDDING_MODEL = "text-embedding-3-small"

datafile_path = "fine_food_reviews_with_embeddings_1k.csv"

df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)

# convert 5-star rating to binary sentiment
df = df[df.Score != 3]
df["sentiment"] = df.Score.replace({1: "negative", 2: "negative", 4: "positive", 5: "positive"})

from utils.embeddings_utils import cosine_similarity, get_embedding

def evaluate_embeddings_approach(
    labels = ['negative', 'positive'],
    model = EMBEDDING_MODEL,
):
    label_embeddings = [get_embedding(label, model=model) for label in labels]

    def label_score(review_embedding, label_embeddings):
        return cosine_similarity(review_embedding, label_embeddings[1]) - cosine_similarity(review_embedding, label_embeddings[0])

    probas = df["embedding"].apply(lambda x: label_score(x, label_embeddings))
    preds = probas.apply(lambda x: 'positive' if x > 0 else 'negative')

    report = classification_report(df.sentiment, preds)
    print(report)

    # Assuming 'df' contains a 'sentiment' column with true labels
    true_labels = (df.sentiment == 'positive').astype(int)  # Convert to binary labels (0 or 1)

    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(true_labels, probas)

    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='*')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

evaluate_embeddings_approach(labels=['negative', 'positive'], model=EMBEDDING_MODEL)
