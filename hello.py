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
    """Generator function to stream response."""
    for word in response_text.split():
        yield word + " "
        time.sleep(0.02)  # Simulate streaming delay

def format_response(response_text):
    """Format response with bullet points if applicable."""
    if response_text.strip().startswith("1."):
        lines = response_text.split(" ")
        formatted_text = "\n".join([f"- {line.strip()}" for line in response_text.split(" ") if line.strip()])
        return formatted_text
    return response_text.replace("\n", "\n\n")

load_dotenv()

st.title("PDF Chatbot with Gemini")

# Sidebar for file upload
st.sidebar.title("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
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
                    formatted_response = format_response(response)
                    st.session_state.chat_history.append({"question": user_input, "response": formatted_response})
                    
                    # Stream response with better formatting
                    with st.chat_message("assistant"):
                        response_placeholder = st.empty()
                        streamed_response = ""
                        for word in stream_response(formatted_response):
                            streamed_response += word
                            response_placeholder.markdown(streamed_response)
                except Exception as e:
                    st.error(f"Error during query: {e}")
    else:
        st.info("Please upload a PDF file to begin.")
