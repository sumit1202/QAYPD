import os
import sys

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

import constants


print(r'''
╔═╗ ╔═╗╦ ╦╔═╗╔╦╗       ╔═╗ ┬ ┬┌─┐┌─┐┌┬┐┬┌─┐┌┐┌   ┬   ╔═╗┌┐┌┌─┐┬ ┬┌─┐┬─┐
║═╬╗╠═╣╚╦╝╠═╝ ║║  ───  ║═╬╗│ │├┤ └─┐ │ ││ ││││  ┌┼─  ╠═╣│││└─┐│││├┤ ├┬┘
╚═╝╚╩ ╩ ╩ ╩  ═╩╝       ╚═╝╚└─┘└─┘└─┘ ┴ ┴└─┘┘└┘  └┘   ╩ ╩┘└┘└─┘└┴┘└─┘┴└─
╦ ╦┌─┐┬ ┬┬─┐  ╔═╗┌─┐┬─┐┌─┐┌─┐┌┐┌┌─┐┬    ╔╦╗┌─┐┌─┐┬ ┬┌┬┐┌─┐┌┐┌┌┬┐┌─┐    
╚╦╝│ ││ │├┬┘  ╠═╝├┤ ├┬┘└─┐│ ││││├─┤│     ║║│ ││  │ ││││├┤ │││ │ └─┐    
 ╩ └─┘└─┘┴└─  ╩  └─┘┴└─└─┘└─┘┘└┘┴ ┴┴─┘  ═╩╝└─┘└─┘└─┘┴ ┴└─┘┘└┘ ┴ └─┘    

- QAYPD => Question-Answer Your Personal Documents''')

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Enable to save to disk & reuse the model (for repeated queries on the same data)
# PERSIST = False
PERSIST = True

query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
  #loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
  loader = DirectoryLoader("data/")
  if PERSIST:
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
  else:
    index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
  # llm=ChatOpenAI(model="gpt-3.5-turbo"),
  # llm=ChatOpenAI(model="gpt-4"),
  llm=ChatOpenAI(model_name="gpt-4"),
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []
while True:
  if not query:
    query = input("\n----------------------------------------------------------------------------------------------------------------\n\nPROMPT: ")
  if query in ['quit', 'q', 'exit']:
    print('exited.')
    sys.exit()
  result = chain({"question": query, "chat_history": chat_history})
  print("GAIN: "+result['answer'])

  chat_history.append((query, result['answer']))
  query = None