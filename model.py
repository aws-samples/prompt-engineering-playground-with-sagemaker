import streamlit as st

from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import VectorDBQA

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import Chroma, AtlasDB, FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders.csv_loader import CSVLoader


from langchain.prompts import PromptTemplate

import json
import os
from datetime import datetime
import jsonlines


import time
import sagemaker, boto3, json
from sagemaker.session import Session
from sagemaker.model import Model
from sagemaker import image_uris, model_uris, script_uris, hyperparameters
from sagemaker.predictor import Predictor
from sagemaker.utils import name_from_base
from typing import Any, Dict, List, Optional
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.llms.sagemaker_endpoint import ContentHandlerBase
import sagemaker

import logging
logger = logging.getLogger('streamlit')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

aws_region="us-east-1"
embedding_endpoint_name="jumpstart-example-raglc-huggingface-tex-2023-05-18-20-02-56-605"
llm_endpoint_name="j2-grande-instruct"

from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler

class SagemakerEndpointEmbeddingsJumpStart(SagemakerEndpointEmbeddings):
    def embed_documents(self, texts: List[str], chunk_size: int = 5) -> List[List[float]]:
        """Compute doc embeddings using a SageMaker Inference Endpoint.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size defines how many input texts will
                be grouped together as request. If None, will use the
                chunk size specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        results = []
        _chunk_size = len(texts) if chunk_size > len(texts) else chunk_size

        for i in range(0, len(texts), _chunk_size):
            response = self._embedding_func(texts[i : i + _chunk_size])
            print
            results.extend(response)
        return results


class ContentHandlerEmb(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs={}) -> bytes:
        input_str = json.dumps({"text_inputs": prompt, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        embeddings = response_json["embedding"]
        return embeddings


from langchain.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint

parameters = {
    "max_length": 200,
    "num_return_sequences": 1,
    "top_k": 250,
    "top_p": 0.95,
    "do_sample": False,
    "temperature": 1,
}

parameters_ai21 = {
    "maxTokens": 200,
    "temperature": 0,
    "numResults": 1,
}


class ContentHandlerLlm(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs={}) -> bytes:
        input_str = json.dumps({"text_inputs": prompt, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["generated_texts"][0]

    
class ContentHandlerLlmAI21(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs={}) -> bytes:
        input_str = json.dumps({"prompt": prompt, **model_kwargs})
        #logging.info(input_str)
        
        #print(input_str)
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        #print(response_json)
        return response_json["completions"][0]["data"]["text"]

prompt_template = """Answer based on context:\n\n{context}\n\n{question}"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


content_handler_emb = ContentHandlerEmb()

embeddings = SagemakerEndpointEmbeddingsJumpStart(
    endpoint_name=embedding_endpoint_name,
    region_name=aws_region,
    content_handler=content_handler_emb,
)

loader = CSVLoader(file_path="docs/processed_data.csv")
#from langchain.document_loaders import UnstructuredHTMLLoader
#loader = UnstructuredHTMLLoader("docs/processed_data.html")

documents = loader.load()

#index_creator = VectorstoreIndexCreator(
# vectorstore_cls=FAISS,
# embedding=embeddings,
#  text_splitter=CharacterTextSplitter(chunk_size=300, chunk_overlap=0),
#)

#index = index_creator.from_loaders([loader])

docsearch = FAISS.from_documents(documents, embeddings)

@st.cache_resource
def get_chain():
    """Read the vector database from disk, build the chain and return it, cached."""
    
    content_handler_llm = ContentHandlerLlm()
    content_handler_ll_ai21 = ContentHandlerLlmAI21()
    
    #sm_llm = SagemakerEndpoint(
    #    endpoint_name="jumpstart-example-raglc-huggingface-tex-2023-05-18-19-55-53-254",
    #    region_name=aws_region,
    #    model_kwargs=parameters,
    #    content_handler=content_handler_llm,
   # )
    
    sm_llm_ai21 = SagemakerEndpoint(
        endpoint_name=llm_endpoint_name,
        region_name=aws_region,
        model_kwargs=parameters_ai21,
        content_handler=content_handler_ll_ai21,
    )
    

    #chain = load_qa_chain(llm=sm_llm, prompt=PROMPT)
    chain = load_qa_chain(llm=sm_llm_ai21, prompt=PROMPT)
    
    return chain

def call_chain(chain, prompt, user_id):
    "Call the chain for a given prompt and user_id, log the results."
    
    docs = docsearch.similarity_search(prompt, k=10)
    #print(f"Found: {docs}")
    
    reply = chain({"input_documents": docs, "question": prompt}, return_only_outputs=False)
    
    raw_result = reply[
        "output_text"
    ]
    
    
    result = {
        'question': prompt,
        'feedback': None,
        'model': 'ai21',
        'answer': raw_result,
       'sources': ["test"]

    }
    log_engagement(result, user_id)
    #print(raw_result)
    return result


def log_engagement(result: dict, user_id: str, path="logs/"):
    data = {
         "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         "user_id": user_id,
         "result": result,
    }

    # Create the logs directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    file_location=f"{path}log-{user_id}.jsonl"

    # Open the user specific log file in append and write the new entry
    with open(file_location, "a") as file:
        file.write(json.dumps(data))
        file.write('\n')


def update_log_engagement_with_feedback(user_id: str, i: int, feedback: int, path="logs/"):

    file_location=f"{path}log-{user_id}.jsonl"
    # Open the file for reading and load its contents as a list of JSON objects
    with jsonlines.open(file_location, 'r') as file:
        data = list(file)
    # Update the best_answer value on the specified line
    data[i]['result']["feedback"] = feedback

    # Save the updated JSON object back to the file
    with jsonlines.open(file_location, 'w') as file:
        file.write_all(data)