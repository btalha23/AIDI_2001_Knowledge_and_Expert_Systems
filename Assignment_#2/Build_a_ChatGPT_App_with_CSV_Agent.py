# Build an LLM App using LangChain

# pip install --quiet streamlit
# pip install --quiet openai
# pip install --quiet langchain
# pip install --quiet langchain_community

import streamlit as st
# from langchain.llms import OpenAI # -> Importing LLMs from langchain is deprecated. 
                                    # -> Importing from langchain will no longer be supported as of langchain==0.2.0.
from langchain_community.llms import OpenAI
from langchain.chat_models import ChatOpenAI
# from langchain_openai import ChatOpenAI

from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
import pandas as pd

openai_api_key = 'OPENAI_API_KEY'

def generate_response(input_text):

    csv_agent = create_csv_agent(ChatOpenAI(temperature=0,
                                            model="gpt-3.5-turbo-0613"), 
                                'titanic.csv',
                                verbose=True,
                                agent_type=AgentType.OPENAI_FUNCTIONS,)
    # csv_agent.run('What are mortality rates considering passenger class and gender? which category had best mortality?')
    csv_agent.run(input_text)
# generate_response('Hello World')

text = input('Question: ')
generate_response(text)

