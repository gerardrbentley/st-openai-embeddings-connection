import streamlit as st
from st_openai_embeddings_connection import OpenAIEmbeddingsConnection

from helpers import plot_distance_matrix, encode_text

st.header("Embedding Connection App")
with st.expander("Show Details"):
    with st.echo():
        conn = st.experimental_connection(
            "openai_embeddings", type=OpenAIEmbeddingsConnection
        )
        st.help(conn)
        st.help(conn.query)

    st.subheader("Cursor Info")
    with st.echo():
        c = conn.cursor()
        st.markdown(type(c))

st.subheader("Embedding Demo")
single_text = "Single Text"
multi_text = "Multiple Texts"
single_text_tokens = "Single Text as Tokens (output should match Single Text)"
multi_text_tokens = "Multiple Texts as Tokens (output should match Multiple Texts)"

choice = st.radio(
    "Demo Type", (multi_text, single_text, multi_text_tokens, single_text_tokens)
)
multi_example = """\
Kittens are great
Kittens are evil
That person is great
This food is great
Kittens are not great
Kittens are awesome
"""
if choice == single_text:
    with st.form("text"):
        text_to_embed = st.text_input(
            "Text To Embed. The full text entry will be embedded.", "Puppies are good"
        )
        submitted = st.form_submit_button()

    if not submitted:
        st.warning("Submit an entry to continue")
        st.stop()

    result = conn.query(text_to_embed)
    result
elif choice == multi_text:
    with st.form("text"):
        texts_to_embed = st.text_area(
            "Texts To Embed. Each new line will be treated as an entry to be embedded.",
            multi_example,
        )
        submitted = st.form_submit_button()

    if not submitted:
        st.warning("Submit an entry to continue")
        st.stop()

    clean_texts = [x for x in texts_to_embed.split("\n") if x != ""]
    result = conn.query(clean_texts)
    embeddings = [result[x].tolist() for x in result.columns]
    plot_distance_matrix(clean_texts, embeddings)
    result
elif choice == single_text_tokens:
    with st.form("text"):
        text_to_embed = st.text_input(
            "Text To Embed. The full text entry will be embedded.", "Puppies are good"
        )
        submitted = st.form_submit_button()

    if not submitted:
        st.warning("Submit an entry to continue")
        st.stop()

    result = conn.query(encode_text(text_to_embed))
    result
elif choice == multi_text_tokens:
    with st.form("text"):
        texts_to_embed = st.text_area(
            "Texts To Embed. Each new line will be treated as an entry to be embedded.",
            multi_example,
        )
        submitted = st.form_submit_button()

    if not submitted:
        st.warning("Submit an entry to continue")
        st.stop()

    clean_texts = [x for x in texts_to_embed.split("\n") if x != ""]
    encoded_texts = [encode_text(x) for x in clean_texts]
    result = conn.query(encoded_texts)
    embeddings = [result[x].tolist() for x in result.columns]
    plot_distance_matrix(clean_texts, embeddings)
    result
