# -*- coding: utf-8 -*-
"""Elina_PDF_Based_RAG_Pipeline.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uxV083JqCTJdil-g88VoODFGMCqZNzTB
"""

! pip install -q pyPDF2 openai pinecone-client langchain_core langchain_openai
! pip install -q langchain-text-splitters

"""### Read the texts in text file and preprocess"""

# read the text file
with open("/content/SNAP correct final Samples.txt", "r") as t_file:
    data = t_file.read()

# processed the text file
# question and answers all mist be together
# Splitting does not make sense here
text_data = data.split("\n\n")
preprocessed_text = [t_data.replace("\n", "").replace("*", "").strip() for t_data in text_data if (t_data != "" and t_data != " ")]

"""### Text Splitting and Embedding Creation and Embedding Storing

### Setting up openai
"""

from openai import OpenAI
import os


os.environ["OPENAI_API_KEY"] = ""
TEXT_MODEL = "text-embedding-ada-002"

# create client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def get_openai_embeddings(text: str) -> list[float]:
    response = client.embeddings.create(input=f"{text}", model=TEXT_MODEL)

    return response.data[0].embedding

"""#### Setting up Pinecone"""

import os
import uuid
from pinecone import Pinecone

NAMESPACE_KEY = "Elina"
os.environ["PINECONE_API_KEY"] = ""
os.environ["INDEX_HOST"] = ""

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(host=os.environ["INDEX_HOST"])

# function to store embeddings
def upsert_embeddings(embeddings, meta_data, namespace_ = NAMESPACE_KEY):
    vector_id = str(uuid.uuid4())
    upsert_response = index.upsert(
    vectors=[
        (vector_id, embeddings, meta_data),
    ],
    namespace=namespace_
    )

    return upsert_response


# function query similar chunks
def query_response(query_embedding, k = 1, namespace_ = NAMESPACE_KEY):
    query_response = index.query(
        namespace=namespace_,
        vector=query_embedding,
        top_k=k,
        include_values=False,
        include_metadata=True,
    )

    return query_response


def content_extractor(similar_data):
    top_values = similar_data["matches"]
    # get the text out
    text_content = [sub_content["metadata"]["text"] for sub_content in top_values]
    return " ".join(text_content)

"""### Question answering pipeline"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


QA_MODEL = "gpt-3.5-turbo"
COMMON_TEMPLATE = """
"Use the following pieces of context to answer the question at the end with human readable answer as a paragraph"
"Please do not use data outside the context to answer any questions. "
"If the answer is not in the given context, just say that you don't have enough context."
"don't try to make up an answer. "
"\n\n"
{context}
"\n\n"
Question: {question}
"n"
"Helpful answer:   "
"""

def get_model():
    model = ChatOpenAI(model=QA_MODEL, api_key=os.environ["OPENAI_API_KEY"])
    return model


def question_answering(query_question: str, context_text: str, template: str = COMMON_TEMPLATE):
    prompt = ChatPromptTemplate.from_template(template)
    model = get_model()
    output_parser = StrOutputParser()

    # create the chain
    chain = prompt | model | output_parser

    # get the answer
    answer = chain.invoke({"context": context_text, "question": query_question})

    return answer

"""### Start Uploading the Files to Pinecone"""

for chunk_text_data in preprocessed_text:
    # get embeddings
    embeddings = get_openai_embeddings(chunk_text_data)
    # create metadata
    meta_creator = {"text": chunk_text_data}
    # store embeddings
    emb_response = upsert_embeddings(embeddings, meta_creator)

"""### Search Query"""

QUERY = "What to do if I can't understand my student?"

# get the query embeddings
quer_embed_data = get_openai_embeddings(QUERY)

# query the similar chunks
similar_chunks = query_response(quer_embed_data)

# extract the similar text data
similar_content = content_extractor(similar_chunks)

# get the answer
answer = question_answering(QUERY, similar_content)

answer