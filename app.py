import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
import os
from dotenv import load_dotenv
import tempfile
import time  # Added missing import

def stream_response(response_text):
    """Generator function to stream response with proper formatting."""
    formatted_text = format_response(response_text)
    streamed_text = ""
    for char in formatted_text:
        streamed_text += char
        yield streamed_text  # Stream the progressively formatted response
        time.sleep(0.01)  # Simulate streaming delay

def format_response(response_text):
    """Format response to have each item on a new line, ensuring proper spacing."""
    items = response_text.split(" ")  # Split based on spaces
    return "\n".join(items)  # Join with new lines

load_dotenv()
# USE this in local enviroments
#api_key= os.getenv("GOOGLE_API_KEY")

st.title("PDF Chatbot with Gemini")

# Sidebar for file upload
st.sidebar.title("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
api_key = st.secrets["GOOGLE_API_KEY"]
llm = GoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05", google_api_key=api_key)
memory = ConversationBufferWindowMemory(k=5)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file is not None:
    st.sidebar.success("Uploaded successfully!")
    
    # Load PDF using tempfile
    with st.spinner("Loading PDF..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
            os.remove(temp_file_path)
        except Exception as e:
            st.error(f"Error loading PDF: {e}")
            loader = None
    
    if loader:
        with st.spinner("Creating Embedding..."):
            embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            index_creator = VectorstoreIndexCreator(
                embedding=embedding,
                text_splitter=text_splitter
            )
            store = index_creator.from_documents(documents)

        st.title("Chat with your PDF")

        # Display chat history using chat-style UI
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(f"**You:** {chat['question']}")
            with st.chat_message("assistant"):
                st.markdown(f"**Bot:** {chat['response']}")

        user_input = st.chat_input("Enter your question:")

        if user_input:
            with st.chat_message("user"):
                st.markdown(f"**You:** {user_input}")
            with st.spinner("Generating response..."):
                try:
                    response = store.query(user_input, llm=llm, memory=memory)
                    st.session_state.chat_history.append({"question": user_input, "response": response})
                    
                    # Stream response with better formatting in real-time
                    with st.chat_message("assistant"):
                        response_placeholder = st.empty()
                        for streamed_text in stream_response(response):
                            response_placeholder.markdown(streamed_text)
                except Exception as e:
                    st.error(f"Error during query: {e}")
    else:
        st.info("Please upload a PDF file to begin.")
