# Build a ChatGPT-like App to talk to CSV files using LangChain

# Python libraries to be installed
# pip install --quiet streamlit
# pip install --quiet openai
# pip install --quiet langchain
# pip install --quiet langchain_community

# Importing the required libraries & modules
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType

# Getting the OPENAI API Key from the user in the app to protect the API key to be misused
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')


def generate_response_from_csv(input_csv, input_text):
    '''
    Function to interact with the CSV file and generate a response to the query/ user prompt

        Parameters:
            input_csv: info of the input CSV file
            input_text: user prompt/ user query
    '''

    # Initialize the lanuage model
    llm = ChatOpenAI(temperature=0, 
                     model="gpt-3.5-turbo-0613")
    
    # Prepare the CSV agent
    csv_agent = create_csv_agent(llm, 
                                input_csv, #'titanic.csv',
                                verbose=True,
                                agent_type=AgentType.OPENAI_FUNCTIONS,
                                )

    # Check if there is a valid user prompt available
    if input_text is not None and input_text != "":
        print(f"The input text coming from the form is: {input_text}")
        # Invoke the agent to generate the response
        response = csv_agent.run(input_text)
        response
    else:
        st.info(body='Invalid user input')


def main():
    st.title('🦜🔗 CSV 📈 Interact App')

    # Get the CSV file from the user
    user_csv = st.file_uploader(label="Upload a CSV file",
                                type="csv")
    
    # Check the validity of the CSV file to proceed further in the program
    if user_csv is not None:
        # Create a form for inputting the user prompt and displaying the response
        with st.form('my_form'):
            # Get the user prompt/ user query (input_text)
            text = st.text_area('Enter a Question/ Thought Regarding the Uploaded CSV:', '')
            submitted = st.form_submit_button('Submit')

            # If the OPENAI API KEY has not been provided, display a waring message
            if not openai_api_key.startswith('sk-'):
                st.warning('Please enter your OpenAI API key!', icon='⚠')
            
            # If the user has input the prompt as well as a valid OPENAI API KEY, 
            # the interaction with the CSV file can start
            if submitted and openai_api_key.startswith('sk-'):
                print(f"The submitted text is: {text}")
                generate_response_from_csv(input_csv=user_csv,input_text=text.strip())


if __name__ == "__main__":
    main()

