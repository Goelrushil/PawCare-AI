import os
from flask import Flask, render_template, request, jsonify
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings

import torch
from transformers import AutoModel, AutoTokenizer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load HuggingFace model and tokenizer
model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": str(device)}
)

# Load and process documents
loader = DirectoryLoader("./data/", glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

# Create FAISS vector store
vector_store = FAISS.from_documents(text_chunks, embeddings)

# Initialize Ollama (make sure Ollama is installed and set up)
llm = Ollama(model="llama3")  # Change model if needed (e.g., "llama3" is just a placeholder)

# Setup memory for conversation context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create ConversationalRetrievalChain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    memory=memory,
)

# Initialize Flask app
app = Flask(__name__)

# Home page route
@app.route("/")
def index():
    return render_template("index.html")  # Make sure you have an HTML template

# Chat route (receives JSON and returns JSON)
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()  # Get JSON data from frontend
    user_input = data.get("user_input")
    if not user_input:
        return jsonify({"response": "Please provide a valid question."}), 400
    
    # Get response from Ollama using ConversationalRetrievalChain
    result = chain({"question": user_input, "chat_history": []})
    
    # Return the AI response
    return jsonify({"response": result["answer"]})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
