import streamlit as st
import pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent, create_retriever_tool
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Qdrant, Pinecone, FAISS
from langchain.callbacks import StreamlitCallbackHandler, StreamingStdOutCallbackHandler
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents import AgentExecutor
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
import qdrant_client
import firebase_admin
from firebase_admin import credentials
import langchain
import openai
openai.api_key = st.secrets["OPENAI_API_KEY"]

from contants import QDRANT_API_KEY, QDRANT_URL, QA_SYSTEM_PROMPT, G_CERT
from tools.firebase_client import FirebaseClient, RecommendVideoTool
from tools.vectorstore import MyVectorStoreQAWithSourcesTool
import os
import shutil
langchain.verbose = True

def get_vectorstore(uploaded_files):
    # text = ""
    # for pdf in pdf_docs:
    #     pdf_reader = PdfReader(pdf)
    #     for page in pdf_reader.pages:
    #         text += page.extract_text()
    # return text
    if os.path.exists("data"):
        shutil.rmtree("data")
    os.makedirs("data")
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        filename = os.path.join("data", uploaded_file.name)  # Path to save the file in the "data" directory

         # Save the file in the "data" directory
        with open(filename, "wb") as f:
            f.write(bytes_data)

        st.sidebar.write(f"File '{uploaded_file.name}' saved in the 'data' directory.")
    loader = PyPDFDirectoryLoader('data')
    all_docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = splitter.split_documents(all_docs)
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

    return vectorstore


# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks


# def get_vectorstore(text_chunks):
#     embeddings = OpenAIEmbeddings()
#     # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore




@st.cache_resource
def init_db():
    cred = credentials.Certificate(G_CERT)
    firebase_admin.initialize_app(cred)
    firebase_cli = FirebaseClient()
    # q_cli = qdrant_client.QdrantClient(
    #     url=QDRANT_URL, api_key=QDRANT_API_KEY)
    # boa_db = Qdrant(q_cli, "boa_documents", OpenAIEmbeddings())
    return firebase_cli


@st.cache_resource
def init(model_name, system_prompt) -> AgentExecutor:
    firebase_cli = init_db()
    v_doc = st.session_state['v_doc']
    llm = ChatOpenAI(temperature=0, model=model_name)

    pinecone.init(
        api_key='07134ff5-efff-48bc-9910-37cdf1eaf687',
        environment='us-central1-gcp'
    )
    amotions_db = Pinecone.from_existing_index(
        'amotions-data-index', OpenAIEmbeddings())

    doc_desc = MyVectorStoreQAWithSourcesTool.get_description(
        "search_documents", "information about documents uploaded")
    amotions_general_desc = MyVectorStoreQAWithSourcesTool.get_description(
        "amotions_general", "information regarding user's feelings or emotions")

    tools = [
        MyVectorStoreQAWithSourcesTool(
            name="search_documents", description=doc_desc, verbose=True, vectorstore=v_doc, llm=llm, search_keywords={'k': 3}),
        MyVectorStoreQAWithSourcesTool(
            name="amotions_general", description=amotions_general_desc, verbose=True, vectorstore=amotions_db, llm=llm, search_keywords={'k': 3}),
        RecommendVideoTool(client=firebase_cli)
    ]
    agent_executor = create_conversational_retrieval_agent(
        llm, tools, verbose=True, system_message=SystemMessage(content=system_prompt))
    return agent_executor


st.set_page_config(page_title="Amotions Demo",
                   page_icon="https://www.amotionsinc.com/navbar-logo.svg")
st.image("https://www.amotionsinc.com/navbar-logo.svg")
st.title("Amotions demo")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'v_doc' not in st.session_state:
    st.session_state['v_doc'] = None

with st.sidebar:
    st.subheader("Your documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing"):
            # # get pdf text
            # raw_text = get_pdf_text(pdf_docs)

            # # get the text chunks
            # text_chunks = get_text_chunks(raw_text)

            # create vector store
            vectorstore = get_vectorstore(pdf_docs)

            st.session_state['v_doc'] = vectorstore
    model_name = st.selectbox("GPT model", ['gpt-4', 'gpt-3.5-turbo'])
    system_prompt = st.text_area(
        "System message", value=QA_SYSTEM_PROMPT, height=600)




for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


if st.session_state['v_doc']:
    agent_executor = init(model_name, system_prompt)

    if user_input := st.chat_input("How can I help you?"):
        st.session_state['messages'].append(
            {'role': 'user', 'content': user_input})
        with st.chat_message('user'):
            st.markdown(user_input)
        with st.chat_message('ai'):
            st_cb = StreamlitCallbackHandler(
                st.container(), expand_new_thoughts=False)
            std_cb = StreamingStdOutCallbackHandler()
            message_placeholder = st.empty()
            message_placeholder.markdown("**loading...**")
            resp = ""
            resp = agent_executor({'input': user_input}, callbacks=[])

            message_placeholder.write(resp['output'])
            st.session_state['messages'].append(
                {'role': 'ai', 'content': resp['output']})
else:
    st.info("Upload documents")
