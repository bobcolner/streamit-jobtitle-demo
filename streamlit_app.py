import os
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from docarray import Document, DocumentArray
# https://docs.streamlit.io

@st.cache(persist=True, allow_output_mutation=True)
def fetch_model() -> SentenceTransformer:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model

@st.cache(persist=True, allow_output_mutation=True)
def fetch_embeddings() -> DocumentArray:
    os.system("curl https://sandbox.zenodo.org/record/1134902/files/full_jobtitles_da.bin?download=1 --output full_jobtitles_da.bin")
    da = DocumentArray.load('full_jobtitles_da.bin')
    return da


def show_sim(da: DocumentArray, query: str, model: SentenceTransformer):
    
    query_vec = model.encode(query, convert_to_tensor=True)
    r = da.find(
        query=query_vec,
        limit=10
    )
    for row in r:
        # row.summary()
        # print(row.text)
        st.subheader(row.text)
        st.write(row.tags)

# app UI
st.title('Job-title Sementic Search Demo')
model = fetch_model()
da = fetch_embeddings()
query = st.text_input('Job-title Query: ', 'ML Developer')
show_sim(da, query, model)
