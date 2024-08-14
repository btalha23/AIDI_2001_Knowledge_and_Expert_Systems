"""
A ChatGPT-like App to talk to PDF files using LangChain
    - LangChain Chains -> create_stuff_documents_chain
                       -> create_retrieval_chain
    - Embeddings -> OpenAIEmbeddings
    - Vector dB  -> Chroma
    - LLM -> ChatOpenAI
"""

# Python libraries to be installed
# pip install --quiet streamlit
# pip install --quiet openai
# pip install --quiet langchain
# pip install --quiet langchain_community
# pip install --quiet langchain_chroma 
# pip install --quiet langchain_openai
# pip install --quiet pypdf

# Importing the required libraries & modules
import os
import tempfile
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Path to the directory to save Chroma database
CHROMA_PATH = "chroma"

# Getting the OPENAI API Key from the user in the app to protect the API key to be misused
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

def get_input_pdf_url(input_pdf):
    """
    The LangChain document loader exploited in this app is a PDF loader, which requires the URL
    of the file. This function is creating a temporary URL for the PDF document uploaded from
    the app GUI.

        Parameters:
            input_pdf: streamlit object created as a result of the function st.file_uploader()

        Returns:
            input_pdf_url: the created URL associated with the uploaded PDF file 
    """
    if input_pdf:
        temp_dir = tempfile.mkdtemp()
        input_pdf_url = os.path.join(temp_dir, input_pdf.name)
        with open(input_pdf_url, "wb") as f:
                f.write(input_pdf.getvalue())
    return input_pdf_url

def load_pdf_file_to_rag(input_pdf_url):
    """
    Function for loading the PDF file to the RAG system

        Parameters:
            input_pdf_url: the created URL associated with the uploaded PDF file

        Returns:
            docs: text of the uploaded PDF file 
    """
    loader = PyPDFLoader(input_pdf_url)
    docs = loader.load()
    return docs

def prep_docs_for_rag_retriever(docs):
    """
    Function for preparing the text for the RAG system. It splits the text into chunks, 
    apply embedding to the text, & then finally store the embeddings to the vector dB 

        Parameters:
            docs: text of the uploaded PDF file

        Returns:
        
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, 
                                        embedding=OpenAIEmbeddings(api_key=openai_api_key),
                                        persist_directory=CHROMA_PATH)
    vectorstore.persist()

def create_retriever_for_rag():
    """
    Function for creating the retriever segment of the RAG system 

        Parameters:

        Returns:
        
    """
    # Prepare the database
    vectorstore = Chroma(persist_directory=CHROMA_PATH, 
                         embedding_function=OpenAIEmbeddings(api_key=openai_api_key))

    # Create the retriever object
    retriever = vectorstore.as_retriever()
    return retriever

def generate_prompt(input_text):
    """
    Function to craft the prompt for the RAG system 

        Parameters:
            input_text: user question/ user prompt

        Returns:
        
    """
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", input_text),
        ]
    )
    return prompt

def generate_response_from_pdf(input_pdf, input_text):
    '''
    Function to interact with the PDF file and generate a response to the query/ user prompt

        Parameters:
            input_pdf: info of the input PDF file
            input_text: user prompt/ user query
    '''

    input_pdf_url = get_input_pdf_url(input_pdf)
    docs = load_pdf_file_to_rag(input_pdf_url)
    prep_docs_for_rag_retriever(docs=docs)
    retriever = create_retriever_for_rag()
    prompt = generate_prompt(input_text=input_text)

    # Initialize the lanuage model
    llm = ChatOpenAI(temperature=0, 
                     model="gpt-3.5-turbo-0613",
                     api_key=openai_api_key)
    
    # Form the complete RAG chain for talking to the PDFs
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Check if there is a valid user prompt available
    if input_text is not None and input_text != "":
        print(f"The input text coming from the form is: {input_text}")
        # Invoke the chain to generate the response
        response = rag_chain.invoke({"input": input_text})
        return response["answer"]
    else:
        st.info(body='Invalid user input')
        return ""


def main():
    st.title('🦜🔗 PDF 📈 Interact App')

    # Get the PDF file from the user
    user_pdf = st.file_uploader(label="Upload a PDF file",
                                type="pdf")
    
    # Check the validity of the PDF file to proceed further in the program
    if user_pdf is not None:
        # Create a form for inputting the user prompt and displaying the response
        with st.form('my_form'):
            # Get the user prompt/ user query (input_text)
            text = st.text_area('Enter a Question/ Thought Regarding the Uploaded PDF:', '')
            submitted = st.form_submit_button('Submit')

            # If the OPENAI API KEY has not been provided, display a waring message
            if not openai_api_key.startswith('sk-'):
                st.warning('Please enter your OpenAI API key!', icon='⚠')

            # If the user has input the prompt as well as a valid OPENAI API KEY, 
            # the interaction with the PDF file can start
            if submitted and openai_api_key.startswith('sk-'):
                print(f"The submitted text is: {text}")
                final_reponse = generate_response_from_pdf(input_pdf=user_pdf, input_text=text)
                if final_reponse is not None and final_reponse != "":
                    st.write(final_reponse)


if __name__ == "__main__":
    main()

