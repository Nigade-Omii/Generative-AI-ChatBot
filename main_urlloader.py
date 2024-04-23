import os
import pickle
import time
import langchain_community
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.vectorstores import  FAISS
import os
import json
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

config = r"Resource_config/configuration.json"

# JSON configuration is stored in a file named 'configuration.json'
with open(config, 'r') as file:
    config_data = json.load(file)

url1_load = config_data.get('url1_path')
url2_load = config_data.get('url2_path')
url3_load = config_data.get('url3_path')

api_key= config_data.get('google_api_key')

url_list=[url1_load,url2_load,url3_load]

file_path =  "faiss_store_openai.pkl" 

embeddings = GoogleGenerativeAIEmbeddings(google_api_key=api_key,model="models/embedding-001")
llm = ChatGoogleGenerativeAI(temperature=0.7, max_output_tokens=1000,google_api_key=api_key,model="gemini-1.5-pro-latest")

    # loading data
loader = UnstructuredURLLoader(urls=url_list)
# print("Loader",loader)

data=loader.load()
# print("Data:",data)


# splitting data
text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n' , "\n" , ".", ","],
    chunk_size = 1000,
    chunk_overlap = 30
)
# main_placeholder.text("Text Splitter... Started...✔✔✔")
docs = text_splitter.split_documents(data)


# Create the FAISS index with the correct dimensionality
vectorstore_openai = FAISS.from_documents(documents=docs, embedding=embeddings)


print(vectorstore_openai.index.ntotal)

#save the FAISS index to pickle file
with open(file_path,"wb") as f:
    pickle.dump(vectorstore_openai, f)
query= input("Enter your Query--")
while (query !='exit'):
    
    if query:
        if os.path.exists(file_path):
            with open(file_path,"rb") as f:
                vectorstore = pickle.load(f)
                retriever=vectorstore.as_retriever()
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
                result = chain({"question": query},return_only_outputs=True)

                print("Answer---",result['answer'])
                print("\n Source---",result['sources'])
    print("for exit Type 'Exit'...")
    query= input("Enter your Query--").lower()
    