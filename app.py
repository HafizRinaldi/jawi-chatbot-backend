from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import google.generativeai as genai
import os

app = Flask(__name__)

# --- KONFIGURASI ---
# Ganti dengan API Key Anda
GEMINI_API_KEY = "AIzaSyChyawjayJJep2IuYOMerGwtXdg51TxgJ4" 
genai.configure(api_key=GEMINI_API_KEY)

# --- MUAT MODEL & DATABASE SAAT SERVER START ---
print("Memuat model dan database...")
retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
jawi_index = faiss.read_index('jawi_index.faiss')
with open('documents.json', 'r', encoding='utf-8') as f:
    documents = json.load(f)
generation_model = genai.GenerativeModel('gemini-1.5-flash-latest')
print("Server siap!")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('query')

    if not user_query:
        return jsonify({"error": "Query tidak ditemukan"}), 400

    # 1. Retrieval
    query_embedding = retriever_model.encode([user_query])
    D, I = jawi_index.search(np.array(query_embedding).astype('float32'), k=2)
    retrieved_context = "\n\n".join([documents[i] for i in I[0]])

    # 2. Augmentation
    prompt = f"""
    Anda adalah seorang ahli aksara Jawi yang ramah. Jawab pertanyaan berikut HANYA berdasarkan konteks yang diberikan.
    Konteks:
    ---
    {retrieved_context}
    ---
    Pertanyaan: {user_query}
    Jawaban:
    """

    # 3. Generation
    try:
        response = generation_model.generate_content(prompt)
        return jsonify({"response": response.text})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)