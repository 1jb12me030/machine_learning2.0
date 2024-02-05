import openai
import pandas as pd

# Set your OpenAI API key
api_key = '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'
client = openai.OpenAI(api_key=api_key)

df = pd.read_csv('fine_food_reviews_1k.csv')

# Combine Title and Content
df["combined"] = "Title: " + df.Summary.str.strip() + "; Content: " + df.text.str.strip()

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

df['embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
df.to_csv('embedded_1k_reviews.csv', index=False)
