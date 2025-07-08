import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load DataFrame from CSV
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_filtered_data_fixed_cleaned.csv")  # Replace with your CSV file path
    # Convert title_bert_embeddings column back to numpy arrays
    df["title_bert_embeddings"] = df["title_bert_embeddings"].apply(eval)
    df["title_bert_embeddings"] = df["title_bert_embeddings"].apply(np.array)
    return df

# Load pre-trained SentenceTransformer model
@st.cache_resource
def load_model():
    return SentenceTransformer("bert-base-nli-mean-tokens")

# Semantic Search Function
def semantic_search(query, top_n=5):
    query_embedding = model.encode([query])  # Embed the query
    document_embeddings = np.vstack(df["title_bert_embeddings"].values)
    similarities = cosine_similarity(query_embedding, document_embeddings).flatten()

    # Retrieve the top N most similar documents
    top_indices = np.argsort(similarities)[::-1][:top_n]
    results = df.iloc[top_indices].copy()
    results["similarity_score"] = similarities[top_indices]
    return results

# Streamlit App
st.title("Welcome to Data Analytics Search Query App")

# Load the model and data
df = load_data()
model = load_model()

# Input Query
query = st.text_input("Enter your query please:", "")

# Perform Semantic Search and Display Results
if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a valid query.")
    else:
        st.info("Searching for relevant results...")
        results = semantic_search(query, top_n=5)

        if not results.empty:
            st.success(f"Found {len(results)} result(s):")
            for _, row in results.iterrows():
                st.subheader(row['cleaned_title'])
                st.write(f"**Authors**: {row['authors']}")
                st.write(f"**Number of Authors**: {row['num_authors']}")
                st.write(f"**Category**: {row.get('primary_category', 'N/A')}")
                st.write(f"**Comment**: {row.get('comment', 'N/A')}")
                st.write(f"**Entry ID**: {row['entry_id']}")
                st.write(f"**Journal Name**: {row.get('journal_name', 'N/A')}")
                st.write(f"**Summary**: {row.get('summary', 'N/A')}")
                st.write(f"**Published Year**: {row.get('published_year', 'N/A')}")
                st.write("---")
        else:
            st.warning("No results found.")