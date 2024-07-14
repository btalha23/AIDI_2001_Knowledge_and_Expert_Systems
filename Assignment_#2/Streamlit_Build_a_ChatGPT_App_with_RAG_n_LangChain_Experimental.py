# Build an LLM App using LangChain

# pip install --quiet streamlit
# pip install --quiet openai
# pip install --quiet langchain
# pip install --quiet langchain_community

import streamlit as st
# from langchain.llms import OpenAI # -> Importing LLMs from langchain is deprecated. 
                                    # -> Importing from langchain will no longer be supported as of langchain==0.2.0.
from langchain_community.llms import OpenAI
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
import pandas as pd

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')   # ''

def generate_response_from_csv(input_csv, input_text):
    llm = ChatOpenAI(temperature=0, 
                     model="gpt-3.5-turbo-0613")
    csv_agent = create_csv_agent(llm, 
                                input_csv, #'titanic.csv',
                                verbose=True,
                                agent_type=AgentType.OPENAI_FUNCTIONS,
                                )

    if input_text is not None and input_text != "":
        print(f"The input text coming from the form is: {input_text}")
        # response = csv_agent.run('What are mortality rates considering passenger class and gender? which category had best mortality?')
        response = csv_agent.run(input_text)
        # print(csv_agent)
        # st.info(response)
        response
    else:
        st.info(body='Invalid user input')


def main():
    st.title('🦜🔗 CSV 📈 Interact App')

    user_csv = st.file_uploader(label="Upload a CSV file",
                                type="csv")
    
    if user_csv is not None:
        with st.form('my_form'):
            text = st.text_area('Enter a Question/ Thought Regarding the Uploaded CSV:', '')
            submitted = st.form_submit_button('Submit')
            if not openai_api_key.startswith('sk-'):
                st.warning('Please enter your OpenAI API key!', icon='⚠')
            if submitted and openai_api_key.startswith('sk-'):
                print(f"The submitted text is: {text}")
                generate_response_from_csv(input_csv=user_csv,input_text=text.strip())


if __name__ == "__main__":
    main()

