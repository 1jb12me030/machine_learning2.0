
import pandas as pd
import tiktoken
import openai
from utils.embeddings_utils import get_embedding
api_key = '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'
embedding_model = "text-embedding-3-small"
embedding_encoding = "cl100k_base"
max_tokens = 8000  # 

input_datapath = "fine_food_reviews_1k.csv"  # to save space, we provide a pre-filtered dataset
df = pd.read_csv(input_datapath, index_col=0)
df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
df = df.dropna()
df["combined"] = (
    "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
)    #The purpose of using strip() here is to ensure that any unwanted whitespaces at the beginning or end of the text in the "Summary" and "Text" columns are removed. 
df.head(2)
print(df.head(2))

top_n = 1000
df = df.sort_values("Time").tail(top_n * 2)  # first cut to first 2k entries, assuming less than half will be filtered out
df.drop("Time", axis=1, inplace=True)

#This line is removing rows with any missing (NaN) values from the DataFrame

encoding = tiktoken.get_encoding(embedding_encoding)

# omit reviews that are too long to embed
df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
df = df[df.n_tokens <= max_tokens].tail(top_n)
len(df)
print(len(df))
df.to_csv("output.csv")
print(df.head(2))
df.head(2)
# This may take a few minutes
df["embedding"] = df.combined.apply(lambda x: get_embedding(x, model=embedding_model))
df.to_csv("fine_food_reviews_with_embeddings_1k.csv")


# the necessary components from the matplotlib library for creating plots. It looks like you are preparing to use t-Distributed Stochastic Neighbor Embedding (t-SNE) to visualize high-dimensional data in a 2D space.
#The concept behind the fit_transform method in scikit-learn, including t-Distributed Stochastic Neighbor Embedding (t-SNE), involves two key steps: 
    #fitting the model to the data and transforming the data into the lower-dimensional space.
