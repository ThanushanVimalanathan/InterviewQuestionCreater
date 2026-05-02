import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from src.prompt import *

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

def file_processing(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()
    
    question_gen = ""
    for page in data:
        question_gen += page.page_content
        
    splitter_ques_gen = TokenTextSplitter(
       model_name = "gpt-3.5-turbo",
       chunk_size=10000,
       chunk_overlap=200
    )
    
    chunk_ques_gen = splitter_ques_gen.split_text(question_gen)
    document_ques_gen = [Document(page_content = t) for t in chunk_ques_gen ]
    
    
    splitter_ans_gen = TokenTextSplitter(
       model_name = "gpt-3.5-turbo",
       chunk_size=1000,
       chunk_overlap=100
   )
    
    document_answer_gen = splitter_ans_gen.split_documents(
      document_ques_gen
   )
    
    return document_ques_gen,document_answer_gen