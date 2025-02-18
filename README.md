# PDF Chatbot with Gemini

This project implements a PDF chatbot using Streamlit, LangChain, and Google's Gemini large language model. It allows users to upload a PDF document and then ask questions about its content.  The chatbot uses embeddings and vector databases to efficiently find relevant information within the PDF and then uses Gemini to generate answers.  It also features streaming responses for a more interactive user experience.

## Features

*   **PDF Upload:** Users can upload PDF files through a Streamlit interface.
*   **Contextual Answers:** The chatbot uses LangChain to create an index of the PDF content, enabling it to provide relevant and contextual answers to user questions.
*   **Gemini Integration:** Leverages Google's Gemini large language model for generating human-like responses.
*   **Streaming Responses:** Provides streaming responses for a more engaging and interactive user experience.
*   **Chat History (Memory):**  Maintains a short chat history using `ConversationBufferWindowMemory` to provide context for subsequent questions.
*   **Clear Status Messages:** Displays informative messages during PDF loading, embedding creation, and response generation.

## Installation

1.  **Clone the repository (optional):**

    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://www.google.com/search?q=https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)  # Replace with your repo URL
    cd PDF-Chatbot-Gemini  # Or the name of the directory
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv  # Create the virtual environment
    .venv\Scripts\activate  # Activate on Windows
    source .venv/bin/activate  # Activate on macOS/Linux
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt # If you create requirements.txt, otherwise install manually.
    pip install streamlit langchain langchain_community google-generativeai python-dotenv
    ```

4.  **Create a `.env` file:**

    Create a file named `.env` in the project directory and add your Google API key:

    ```
    GOOGLE_API_KEY=YOUR_ACTUAL_API_KEY
    ```

    Replace `YOUR_ACTUAL_API_KEY` with your actual Google API key.  You can obtain an API key from the Google Cloud Console.

## Usage

1.  **Run the Streamlit app:**

    ```bash
    streamlit run hello.py  # Replace hello.py with the actual name of your script
    ```

2.  **Open the app in your browser:**

    Streamlit will provide a URL (usually `http://localhost:8501`) that you can use to access the app in your web browser.

3.  **Upload a PDF:**

    Use the file uploader in the sidebar to select and upload a PDF file.

4.  **Ask questions:**

    Enter your questions in the text box and click "Submit."

5.  **View the responses:**

    The chatbot's responses will be displayed below the text box.  Streaming responses will appear chunk by chunk.

## Project Structure