import streamlit as st
import pandas as pd
import openai
import os
from description import ModelHandler
from typing import List
import numpy as np
from tenacity import retry, wait_random_exponential, stop_after_attempt



def file_walk(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            if not name.endswith(".DS_Store"):  # Skip .DS_Store files
                yield root, name


def get_format(path):
    file_format = "." + path.split(".")[-1]
    if file_format == ".xlsx":
        read_func = pd.read_excel
    elif file_format == ".csv":
        read_func = pd.read_csv
    else:
        raise ValueError(f"File format {file_format} not supported.")
        print("unected file: " + path)
    return file_format, read_func

def embed(file_path, embeddings):
    file_format, read_func = get_format(file_path)

    df = read_func(file_path)
    embedding_path = file_path.replace("describe", "embeddings").replace(
        file_format, ".parquet"
    )


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, engine="text-similarity-davinci-001", use_gemini=False, **kwargs) -> List[float]:

    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    if use_gemini:
        import google.generativeai as genai
        # FIXME: ignores kwargs
        embedding = genai.embed_content(
            model=engine,
            content=text,
            task_type="retrieval_document"
        )["embedding"]
    else:
        embedding = openai.Embedding.create(input=[text], engine=engine, **kwargs)["data"][0]["embedding"]
    return embedding



def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))








class Embeddings:
    def __init__(self):
        self.df_dict = None

    def search2(self, query, top_n=3):
        """
        Search the embeddings for the most similar entries to the query.
        Automatically selects and configures the embedding model via get_model().
        """

        MH = ModelHandler()
        # Get the preferred model and service name
        models = MH.get_model()
        if not models:
            raise RuntimeError("No valid model found. Check .streamlit/secrets.toml configuration.")
    
        model_obj, service_name = models[0]

        # Dynamically decide which embedding engine to use
        if service_name == "gemini":
            # Use Gemini embeddings
            embedding = get_embedding(query, engine=None, use_gemini=True)

        elif service_name == "gpt":
            # Use GPT embeddings
            #embedding = get_embedding(query, engine=None, use_gemini=False)
            embedding= engine or service_conf.get("ENGINE_ADA", "text-embedding-ada-002")

        else:
         raise ValueError(f"Unsupported service: {service_name}")

        # Perform similarity search
        df = self.df_dict.copy()
        df["similarities"] = df.user_embedded.apply(
            lambda x: cosine_similarity(x, embedding)
        )

        # Optional: threshold and top-N filtering
        df = df[df.similarities > 0.7]
        res = df.sort_values("similarities", ascending=False).head(top_n)

        return res

    

    def search(self, query, top_n=3):
        # type is the index into the various dataframes stored in the embeddings.
        # if type is not specified, it will search all dataframes
        # otherwise it will search those listed
        MH = ModelHandler()

        model = MH.get_model()  

        if USE_GEMINI:
            import google.generativeai as genai

            genai.configure(api_key=GEMINI_API_KEY)
            ENGINE = GEMINI_EMBEDDING_MODEL
        else:
            openai.api_base = GPT3_BASE
            openai.api_version = GPT3_VERSION
            openai.api_key = GPT3_KEY
            # text-embedding-ada-002 (Version 2) model
            ENGINE = ENGINE_ADA
        embedding = get_embedding(query, engine=ENGINE, use_gemini=USE_GEMINI)

        # An option for the future is to take the top from each dataframe, so we get a mixture of responses.
        df = self.df_dict
        df["similarities"] = df.user_embedded.apply(
            lambda x: cosine_similarity(x, embedding)
        )
        df = df[df.similarities > 0.7]

        res = df.sort_values("similarities", ascending=False).head(top_n)
        return res

    def compare_strings(self, string1, string2):
        # Use this function to compare two strings or words embeddings
        # Returns co-sine similarilty between the two strings
        ENGINE = GEMINI_EMBEDDING_MODEL if USE_GEMINI else ENGINE_ADA
        embedding1 = get_embedding(string1, engine=ENGINE, use_gemini=USE_GEMINI)
        embedding2 = get_embedding(string2, engine=ENGINE, use_gemini=USE_GEMINI)

        return cosine_similarity(embedding1, embedding2)

    def return_embedding(self, query):
        if USE_GEMINI:
            import google.generativeai as genai

            genai.configure(api_key=GEMINI_API_KEY)
            ENGINE = GEMINI_EMBEDDING_MODEL
        else:
            openai.api_base = GPT3_BASE
            openai.api_version = GPT3_VERSION
            openai.api_key = GPT3_KEY
            # text-embedding-ada-002 (Version 2) model
            ENGINE = ENGINE_ADA
        embedding = get_embedding(query, engine=ENGINE, use_gemini=USE_GEMINI)

        return embedding



class CreateEmbeddings(Embeddings):
    def __init__(self):
        self.df_dict = CreateEmbeddings.get_embeddings()

    def get_embeddings():
        # Gets all embeddings
        df_embeddings_dict = dict()

        files = [
            "QandA",
        ]

        df_embeddings = pd.DataFrame()
        for file in files:
            # Read in
            df_temp = pd.read_parquet(f"data/embeddings/{file}.parquet")
            if "category" not in df_temp:
                df_temp["category"] = None
            if "format" not in df_temp:
                df_temp["format"] = None
            df_temp = df_temp[
                ["user", "assistant", "category", "user_embedded", "format"]
            ]
            df_temp["user_embedded"] = df_temp.user_embedded.apply(eval).to_list()
            df_embeddings = pd.concat([df_embeddings, df_temp], ignore_index=True)

        return df_embeddings
