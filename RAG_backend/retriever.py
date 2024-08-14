import datasets
from datasets import load_dataset
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
# from tqdm.notebook import tqdm
import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt
import os
import google.generativeai as genai
import time

pd.set_option(
    "display.max_colwidth", None
)

# Configure the API key for Google Gemini
genai.configure(api_key="API")

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

EMBEDDIG_MODEL = "thenlper/gte-small"
MARKDOWN_SEPARATORS = ["\n#{1,6} ", "```\n","\n\\*\\*\\*+\n","\n---+\n","\n\n","\n"," ","",]
KNOWLEDGE_VECTOR_DATABASE = None
tokenizer = AutoTokenizer.from_pretrained(EMBEDDIG_MODEL)

# Define the prompt format
# chat_template_str = """
# {%- for message in messages %}
# {{message.role}}: {{message.content}}
# {%- endfor %}
# """

# tokenizer.chat_template = chat_template_str

# prompt_in_chat_format = [
#     {
#         "role": "system",
#         "content": """Using the information contained in the context, give a comprehensive answer to the question.
#         Respond only to the question asked, the response should be concise and relevant to the question.
#         Provide the number of the source document when relevant.
#         If the answer cannot be deduced from the context, do not give an answer.
#         """
#     },
#     {
#         "role": "user",
#         "content": """
#         Context:{context}
#         ---
#         Now here is the question you need to answer.
#         Question:{question}
#         """
#     }
# ]

def split_documents(
    chunk_size:int,
    knowledge_base:List[LangchainDocument],
    tokenizer_name: Optional[str]=EMBEDDIG_MODEL
    ) -> List[LangchainDocument] :
    """
    Split documents into chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        separators=MARKDOWN_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size/10),
        add_start_index = True,
        strip_whitespace=True,
    )

    docs_processed = []

    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    #Removing duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


def train_RAG():
    global KNOWLEDGE_VECTOR_DATABASE
    data_path = "C:/Users/Shiva/Documents/Projects/RAG_backend/extracted_text.txt"
    print(data_path)

    print(os.path.isfile(data_path)) 

    ds = datasets.load_dataset("text",data_files=data_path,split="train")

    RAW_BASE = [
        LangchainDocument(page_content=doc["text"])
        for doc in ds
    ]
    start = time.time()
    docs_processed = split_documents(512,RAW_BASE,tokenizer_name=EMBEDDIG_MODEL)
    end = time.time()
    elapsed_time = end - start
    print(f"Document processing time: {elapsed_time:.4f} seconds")

    start = time.time()
    print("building embedding model started")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDIG_MODEL,
        multi_process=True,
        # model_kwargs={"device": "cuda"},
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("building embedding model completed")
    end = time.time()
    elapsed_time = end - start
    print(f"Building embeddings time: {elapsed_time:.4f} seconds")

    start = time.time()
    print("vector storage embedding model started")
    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        documents=docs_processed,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.COSINE
    )
    print("vector storage embedding model completed")
    end = time.time()
    elapsed_time = end - start
    print(f"Vector storage time: {elapsed_time:.4f} seconds")

def getAnswer(user_question):
    print('Getting answer')
    start = time.time()
    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(
        query=user_question,
        k=3
    )
    print('Retrieved documents')
    end = time.time()
    elapsed_time = end - start
    print(f"Retrieving time: {elapsed_time:.4f} seconds")
    retrieved_docs_text = [
        doc.page_content for doc in retrieved_docs
    ]
    print(retrieved_docs_text)

    # Correctly format the context with no extra spaces inside {}
    context_sup = "\nExtracted documents:\n"
    context_sup += "".join(
        [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)]
    )

    # Directly define the RAG_PROMPT_TEMPLATE
    RAG_PROMPT_TEMPLATE = """
    Using the information contained in the context, give a comprehensive answer to the question.
    Respond only to the question asked, the response should be concise and relevant to the question.
    Provide the number of the source document when relevant.
    If the answer cannot be deduced from the context, do not give an answer.

    Context: {context}
    ---
    Now here is the question you need to answer:
    Question: {question}
    """

    final_prompt = RAG_PROMPT_TEMPLATE.format(context=context_sup, question=user_question)

    # Make the API call to Google Gemini
    response = model.generate_content(final_prompt)
    return response.text
