# Streamlit OpenAI Embedding Connection

[![Releases](https://img.shields.io/pypi/v/st-openai-embeddings-connection)](https://pypi.org/project/st-openai-embeddings-connection/)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://st-embeddings.streamlit.app)

Example app using Streamlit Connections to query OpenAI's ada-002 embedding model.

Results from these queries can be used to perform semantic similarity searches either locally or in a vector database.

Sample usage:

```sh
pip install st-openai-embeddings-connection
```

`streamlit.py`:

```py
import streamlit as st
from st_openai_embeddings_connection import OpenAIEmbeddingsConnection

conn = st.experimental_connection(
    "openai_embeddings", type=OpenAIEmbeddingsConnection
)
text_to_embed = st.text_input(
    "Text To Embed", "Puppies are good"
)

if st.button("embed"):
    result = conn.query(text_to_embed)
    result

```