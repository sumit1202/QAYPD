import os
import sys
import constants
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI


print(r'''
╔═╗ ╔═╗╦ ╦╔═╗╔╦╗       ╔═╗ ┬ ┬┌─┐┌─┐┌┬┐┬┌─┐┌┐┌   ┬   ╔═╗┌┐┌┌─┐┬ ┬┌─┐┬─┐
║═╬╗╠═╣╚╦╝╠═╝ ║║  ───  ║═╬╗│ │├┤ └─┐ │ ││ ││││  ┌┼─  ╠═╣│││└─┐│││├┤ ├┬┘
╚═╝╚╩ ╩ ╩ ╩  ═╩╝       ╚═╝╚└─┘└─┘└─┘ ┴ ┴└─┘┘└┘  └┘   ╩ ╩┘└┘└─┘└┴┘└─┘┴└─
╦ ╦┌─┐┬ ┬┬─┐  ╔═╗┌─┐┬─┐┌─┐┌─┐┌┐┌┌─┐┬    ╔╦╗┌─┐┌─┐┬ ┬┌┬┐┌─┐┌┐┌┌┬┐┌─┐    
╚╦╝│ ││ │├┬┘  ╠═╝├┤ ├┬┘└─┐│ ││││├─┤│     ║║│ ││  │ ││││├┤ │││ │ └─┐    
 ╩ └─┘└─┘┴└─  ╩  └─┘┴└─└─┘└─┘┘└┘┴ ┴┴─┘  ═╩╝└─┘└─┘└─┘┴ ┴└─┘┘└┘ ┴ └─┘    

- QAYPD => Question-Answer Your Personal Documents''')

os.environ["OPENAI_API_KEY"] = constants.APIKEY

query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

#step1: data loading
#Text File loader  
# loader = TextLoader('data.txt')
#Directory/Folder loader
loader = DirectoryLoader('data/')

#step2: data storing and indexing
index = VectorstoreIndexCreator().from_loaders([loader])

#querying
while True:
  if not query:
    query = input("\n----------------------------------------------------------------------------------------------------------------\n\nPROMPT: ")
  if query in ['quit', 'q', 'exit']:
    print('exited.')
    sys.exit()
  result = index.query(query) #step3: data retrieval and generation
  print("GAIN: "+result)

  query = None
