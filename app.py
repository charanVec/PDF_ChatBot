import os
import uuid
import time
import numpy as np
from flask import Flask, request, render_template, jsonify
import fitz  # PyMuPDF
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from sklearn.metrics.pairwise import cosine_similarity
from chromadb import Client
from chromadb.config import Settings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
key = os.getenv('key')
parser = StrOutputParser()

Model = "llama-3.1-70b-versatile"
model = ChatGroq(api_key=key, model=Model, temperature=0)
llm = model | parser

app = Flask(__name__)

# Initialize ChromaDB client
chroma_client = Client(Settings(persist_directory="./chromadb_storage"))

# Create or load a collection
collection = chroma_client.create_collection(name="pdf_embeddings")

# Global variables
pdf_text_storage = {}
chat_history = []
word2vec_model = None

# Helper function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# Helper function to preprocess text for Gensim
def preprocess_text(text):
    return simple_preprocess(text, deacc=True)

# Helper function to train a Word2Vec model
def train_word2vec_model(text_chunks):
    sentences = [preprocess_text(chunk) for chunk in text_chunks]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)
    return model

# Helper function to get vector representation of a text
def get_vector_representation(text, model):
    tokens = preprocess_text(text)
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# Retry mechanism to handle temporary failures
def ask_chatbot_with_retry(prompt, retries=3, delay=5):
    for i in range(retries):
        try:
            response = llm.invoke(prompt)
            return response
        except Exception as e:
            print(f"Attempt {i+1}/{retries} failed: {e}")
            time.sleep(delay)
    raise Exception("Service Unavailable after retries")

# Route for the homepage
@app.route('/')
def home():
    uploaded_files = list(pdf_text_storage.keys())
    return render_template('home.html', uploaded_files=uploaded_files)

# Route for the upload page
@app.route('/upload')
def upload_page():
    return render_template('upload.html')

# Route to handle file uploads
@app.route('/upload_files', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({"error": "No files part"}), 400

    files = request.files.getlist('files')
    file_names = []

    global word2vec_model
    all_texts = []

    # Ensure the uploads directory exists
    upload_dir = 'uploads'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    for file in files:
        if file.filename == '':
            continue

        if file and file.filename.lower().endswith('.pdf'):
            original_filename = file.filename
            file_path = os.path.join(upload_dir, original_filename)
            file.save(file_path)
            text = extract_text_from_pdf(file_path)
            pdf_text_storage[original_filename] = text
            all_texts.append(text)
            file_names.append(original_filename)
        else:
            return jsonify({"error": "Invalid file type"}), 400

    # Process and store vectors
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = [chunk for text in all_texts for chunk in splitter.split_text(text)]
    word2vec_model = train_word2vec_model(text_chunks)
    text_vectors = np.array([get_vector_representation(chunk, word2vec_model) for chunk in text_chunks])

    # Add vectors and texts to ChromaDB
    collection.add(
        embeddings=text_vectors.tolist(),
        metadatas=[{'chunk': chunk} for chunk in text_chunks],
        ids=[str(uuid.uuid4()) for _ in text_chunks]
    )

    return jsonify({"message": "Files uploaded and text extracted", "file_names": file_names}), 200

# Route to cancel a specific file upload
@app.route('/cancel/<filename>', methods=['DELETE'])
def cancel_file(filename):
    if filename in pdf_text_storage:
        # Remove the PDF text from the storage
        del pdf_text_storage[filename]
        # Delete the uploaded file from the filesystem
        file_path = os.path.join('uploads', filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"message": "File upload canceled"}), 200
    else:
        return jsonify({"error": "File not found"}), 404

# Route for the query page
@app.route('/query')
def query_page():
    return render_template('query.html', chat_history=chat_history)

# Route to handle chatbot queries
@app.route('/ask', methods=['POST'])
def ask_chatbot():
    data = request.json
    question = data.get('question')

    if not question or not pdf_text_storage:
        return jsonify({"error": "Question or PDF text not available"}), 400

    # Vectorize the query
    query_vector = get_vector_representation(question, word2vec_model)

    if query_vector is None or query_vector.size == 0:
        return jsonify({"error": "Unable to vectorize query"}), 400

    # Query the ChromaDB collection for the most similar text chunks
    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=3  # You can adjust the number of results
    )

    # Extract relevant chunks
    relevant_chunks = [metadata['chunk'] for metadata in results['metadatas'][0]]

    # Construct the prompt with the relevant chunks as context
    context = "\n\n".join(relevant_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nProvide a concise and informative answer."

    try:
        response = ask_chatbot_with_retry(prompt)
        chat_history.append({"question": question, "response": response})
        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to clear chat history
@app.route('/clear_history', methods=['POST'])
def clear_history():
    global chat_history
    chat_history.clear()
    return jsonify({"message": "Chat history cleared"}), 200

if __name__ == '__main__':
    app.run(debug=True)
