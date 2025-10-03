# 📄 PDF Q&A Chatbot using LangChain, FAISS & GROQ

This project is a terminal-based chatbot that allows users to ask questions about the contents of a PDF file. It uses LangChain's RAG pipeline, FAISS for vector search, HuggingFace embeddings for semantic understanding, and GROQ's LLaMA 3.1 model for generating responses.

## 🚀 Features

- Load and process any PDF document
- Split content into chunks for efficient retrieval
- Embed using HuggingFace's `all-mpnet-base-v2` model
- Store and search using FAISS vector database
- Query using natural language and get context-aware answers
- Displays response time for performance tracking

## 🧠 Technologies Used

- LangChain
- FAISS
- HuggingFace Embeddings
- GROQ API (LLaMA 3.1)
- Python
- dotenv

## 📁 File Structure

- `raw.py`: Main script
- `.env`: Stores API keys for GROQ and HuggingFace

## 🔐 Environment Variables

Create a `.env` file with the following keys:

GROQ_API_KEY=your_groq_api_key
HUGGINGFACE_API_TOKEN=your_huggingface_token

## 🛠️ How to Run

1. Clone the repository:
```
git clone https://github.com/yourusername/pdf-chatbot.git
cd pdf-chatbot
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Add your .env file with API keys.


4. Run the Streamlit app:
```
streamlit run raw.py
```
