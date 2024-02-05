
import pandas as pd
import openai
from utils.embeddings_utils import get_embedding
# Read CSV file into a DataFrame
df = pd.read_csv('fine_food_reviews_1k.csv')

# Concatenate relevant columns into a single string
#Text = df['Time'].astype(str) + ' ' + df['ProductId'] + ' ' + df['UserId'] + ' ' + df['Score'].astype(str) + ' ' + df['Summary'] + ' ' + df['Text']

# Use openai.Completion.create with input_text
#response = openai.Completion.create(input=input_text, model="text-embedding-ada-002")
#df.to_csv("fine_food_reviews_with_embeddings_1k.csv")
# Process the response as needed
#openai.api_key = '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'

df["embedding"] = df.text.apply(lambda x: get_embedding(x, model="text-embedding-ada-002"))
df.to_csv("fine_food_reviews_with_embeddings_1k.csv")