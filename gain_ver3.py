import os
import sys
import constants
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import SVMRetriever
import logging

from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever


print(r'''
╔═╗ ╔═╗╦ ╦╔═╗╔╦╗       ╔═╗ ┬ ┬┌─┐┌─┐┌┬┐┬┌─┐┌┐┌   ┬   ╔═╗┌┐┌┌─┐┬ ┬┌─┐┬─┐
║═╬╗╠═╣╚╦╝╠═╝ ║║  ───  ║═╬╗│ │├┤ └─┐ │ ││ ││││  ┌┼─  ╠═╣│││└─┐│││├┤ ├┬┘
╚═╝╚╩ ╩ ╩ ╩  ═╩╝       ╚═╝╚└─┘└─┘└─┘ ┴ ┴└─┘┘└┘  └┘   ╩ ╩┘└┘└─┘└┴┘└─┘┴└─
╦ ╦┌─┐┬ ┬┬─┐  ╔═╗┌─┐┬─┐┌─┐┌─┐┌┐┌┌─┐┬    ╔╦╗┌─┐┌─┐┬ ┬┌┬┐┌─┐┌┐┌┌┬┐┌─┐    
╚╦╝│ ││ │├┬┘  ╠═╝├┤ ├┬┘└─┐│ ││││├─┤│     ║║│ ││  │ ││││├┤ │││ │ └─┐    
 ╩ └─┘└─┘┴└─  ╩  └─┘┴└─└─┘└─┘┘└┘┴ ┴┴─┘  ═╩╝└─┘└─┘└─┘┴ ┴└─┘┘└┘ ┴ └─┘    

- QAYPD => Question-Answer Your Personal Documents''')

os.environ["OPENAI_API_KEY"] = constants.APIKEY

question = None
if len(sys.argv) > 1:
  question = sys.argv[1]
  
#Step 1. Load data
loader = DirectoryLoader('data/')
data = loader.load()

#Step 2. Split data
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
all_splits = text_splitter.split_documents(data)

#Step 3. Store data
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

#Step 4. Retrieve and Gnereate data
#Use only the tools provided to look for context to answer the the question at the end. 
#If you don't find the answers in the context or don't know the answers to the user questions, truthfully say you don't know. 
#Don't attempt to make up answers or hallucinate. 
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Never give answer to the the question from out of context. Keep the answer as verbose as possible.
Keep the answer as formal as possible.
Always say "Hope you found what you were looking for!" at the end of the answer and in the next line. 
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

#querying
while True:
  if not question:
    question = input("\n-------------------------------------------------------------\n\nPROMPT: ")
  if question in ['quit', 'q', 'exit']:
    print('exited.')
    sys.exit()
  
  result = qa_chain({"query": question})
  print("\nGAIN: "+result["result"])

  question = None
