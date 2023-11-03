# About QAYPD:
QAYPD(Question-Answer Your Personal Documents) can help you answer questions related to your personal documents swiftly.
It is built using Python and Langchain framework and leverages the computational capabilities of OpenAI's ChatGPT Large Language Model(LLM).

## Installation Guide:

1. Install [Langchain](https://github.com/hwchase17/langchain) and other required packages.
```
pip install langchain openai chromadb tiktoken unstructured
```
2. Update `constants.py` with your your own [OpenAI API key](https://platform.openai.com/account/api-keys).

3. Place your own data into `data/data.txt`. Note: You can add any number of .txt files.

## Few Examples:
Example 1:
```
> python3 gain_ver1.py "Who are you?"
I am an AI-based tool and my name is 'QAYPD' which stands 
for 'Question-Answer Your Personal Documents'.
I can help you with answering your personal documents quickly.
I am built using Python and Langchain framework and leverage the computational capabilities of 
OpenAI's ChatGPT Large Language Model(LLM).
```

Example 2:
```
> python3 gain_ver2.py "Your dog name is?"
My dog's name is Pluto.
```

Example 3:
```
> python3 gain_ver3.py "What is QAYPD?"
'QAYPD' stands 
for 'Question-Answer Your Personal Documents'.
```
