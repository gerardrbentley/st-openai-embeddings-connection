from streamlit.connections import ExperimentalBaseConnection
from streamlit.runtime.caching import cache_data

import requests
import pandas as pd

from typing import Union, List
import os

OPENAI_EMBEDING_URL = "https://api.openai.com/v1/embeddings"
DEFAULT_MODEL = "text-embedding-ada-002"
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = "cl100k_base"

INPUT_KWARG = "input"
MODEL_KWARG = "model"


class OpenAIEmbeddingsConnection(ExperimentalBaseConnection[requests.Session]):
    """st.experimental_connection implementation for Open AI Text Embeddings"""

    def _connect(self, **kwargs) -> requests.Session:
        """set up and return the underlying connection object

        Raises:
            Exception: when "OPENAI_API_KEY" is not available in the application environment

        Returns:
            requests.Session: underlying http session for connection pooling and headers
        """
        if "openai_api_key" in kwargs:
            api_key = kwargs.pop("openai_api_key")
        elif "openai_api_key" in self._secrets:
            api_key = self._secrets["openai_api_key"]
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise Exception("openai_api_key not in kwargs, secrets or environment")
        session = requests.session()
        session.headers["Authorization"] = f"Bearer {api_key}"
        session.headers["Content-Type"] = "application/json"
        return session

    def cursor(self) -> requests.Session:
        """provides http session used in `query`

        Returns:
            requests.Session: openai authenticated http session
        """
        return self._instance

    def query(
        self,
        query: Union[str, List[str], List[float], List[List[float]]],
        ttl: int = 3600,
        **kwargs,
    ) -> pd.DataFrame:
        """given one or more pieces of text,
        attempts to retrieve embeddings from openai ada-002 model,
        returns dataframe with single 'embedding' column for each text

        accepts:
        - a string representing one piece of text to embed
        - an array of floats representing tokens of the text
        - a list of strings representing multiple texts to embed
        - an array of arrays of floats representing tokens of multiple texts

        Args:
            query (Union[str, List[str], List[float], List[List[float]]]): Text to embed
            ttl (int, optional): Duration in seconds to cache results for same input. Defaults to 3600.

        Raises:
            Exception: when results cannot be fetched from openai request

        Returns:
            pd.DataFrame: vector of embedding values
        """

        @cache_data(ttl=ttl)
        def _query(query: str, **kwargs) -> pd.DataFrame:
            body = dict(kwargs)
            if INPUT_KWARG not in body:
                body[INPUT_KWARG] = query
            if MODEL_KWARG not in body:
                body[MODEL_KWARG] = DEFAULT_MODEL

            cursor = self.cursor()
            response = cursor.post(OPENAI_EMBEDING_URL, json=body)

            return handle_embedding_response(response)

        return _query(query, **kwargs)


def handle_embedding_response(response: requests.Response) -> pd.DataFrame:
    """Given an http response from openai embeddings endpoint,
    attempts to parse embedding vectors from each response element,
    returns a dataframe with one column per embedded text

    Args:
        response (requests.Response): openai embeddings response

    Raises:
        Exception: when response does not match expected openai result

    Returns:
        pd.DataFrame: one column per embedded text
    """    
    result_body = response.json()

    data = result_body.get("data")
    if data is None:
        raise Exception("no 'data' attribute in openai embedding response", result_body)

    embeddings = []
    for record in data:
        embedding = record.get("embedding")
        if embedding is None:
            raise Exception(
                "no 'embedding' attribute in openai embedding data response(s)",
                result_body,
            )
        embeddings.append(embedding)

    df = pd.DataFrame(embeddings)
    return df.T
