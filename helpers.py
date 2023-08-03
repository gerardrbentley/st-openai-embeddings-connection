import tiktoken
import streamlit as st
from typing import List
from scipy.spatial.distance import cosine
import pandas as pd
import altair as alt

EMBEDDING_ENCODING = "cl100k_base"


@st.cache_data
def encode_text(text: str) -> List[float]:
    encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
    return encoding.encode(text)


def plot_distance_matrix(strings: List[str], embeddings: List[List[float]]):
    """src: https://github.com/openai/openai-cookbook/blob/main/apps/embeddings-playground/embeddings_playground.py"""
    # create dataframe of embedding distances
    df = pd.DataFrame({"string": strings, "embedding": embeddings})
    df["string"] = df.apply(lambda row: f"{row['string']}", axis=1)
    df["dummy_key"] = 0
    df = pd.merge(df, df, on="dummy_key", suffixes=("_1", "_2")).drop(
        "dummy_key", axis=1
    )
    df = df[df["string_1"] != df["string_2"]]  # filter out diagonal (always 0)
    df["distance"] = df.apply(
        lambda row: cosine(row["embedding_1"], row["embedding_2"]),
        axis=1,
    )
    df["label"] = df["distance"].apply(lambda d: f"{d:.2f}")

    chart_width = (
        50
        + min(256, max(df["string_1"].apply(len) * 8))
        + len(strings) * 80
    )

    # extract chart parameters from data
    color_min = df["distance"].min()
    color_max = 1.5 * df["distance"].max()
    x_order = df["string_1"].values

    # create chart
    boxes = (
        alt.Chart(df, title="ada-002 Embedding Cosine Similarity Results")
        .mark_rect()
        .encode(
            x=alt.X("string_1", title=None, sort=x_order),
            y=alt.Y("string_2", title=None, sort=x_order),
            color=alt.Color(
                "distance:Q",
                title="cosine distance",
                scale=alt.Scale(
                    domain=[color_min, color_max], scheme="darkblue", reverse=True
                ),
            ),
        )
    )

    labels = (
        boxes.mark_text(align="center", baseline="middle", fontSize=32)
        .encode(text="label")
        .configure_axis(labelLimit=256, labelFontSize=16)
        .properties(width=chart_width, height=chart_width)
    )

    st.altair_chart(labels)  # note: layered plots are not supported in streamlit :(

