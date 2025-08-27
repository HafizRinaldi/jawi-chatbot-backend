import json
import os
import numpy as np
import requests
import faiss
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

QWEN_API_URL = "https://litellm.bangka.productionready.xyz/v1/chat/completions"
QWEN_API_KEY = "sk-DP7uvSNB2l8FFbtNCgnPoQ"
MODEL_NAME = "vllm-qwen3"

print("Memuat model retriever (SentenceTransformer)...")
retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Memuat database vektor (FAISS)...")
jawi_index = faiss.read_index('jawi_index.faiss')
print("Memuat dokumen pengetahuan...")
with open('documents.json', 'r', encoding='utf-8') as f:
    documents = json.load(f)
print("✅ Server siap menerima permintaan!")
print("-" * 30)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('query')
    contextual_topic = data.get('context')

    if not user_query: return jsonify({"error": "Query tidak ditemukan"}), 400

    print(f"\n🚀 Menerima query faktual: '{user_query}' | Konteks: '{contextual_topic}'")
    
    affirmative_words = ['iya', 'ya', 'oke', 'boleh', 'ok', 'lanjut', 'jelaskan', 'iyaa']
    if contextual_topic and user_query.lower().strip() in affirmative_words:
        print("💡 Query afirmatif terdeteksi. Mengalihkan query untuk menjelaskan konteks.")
        search_query = contextual_topic
    else:
        search_query = user_query

    query_embedding = retriever_model.encode([search_query])
    distances, indices = jawi_index.search(np.array(query_embedding).astype('float32'), k=3)
    retrieved_context = "\n\n".join([documents[i] for i in indices[0]])
    print(f"📚 Konteks yang relevan ditemukan:\n---\n{retrieved_context}\n---")

    prompt_template = f"""
    Anda adalah JawiAI, seorang ahli aksara Jawi yang cerdas dan presisi.

    IKUTI ATURAN HIERARKI INI:
    1.  **ATURAN SAPAAN:** Jika pengguna menyapa ('hai', 'hallo'), balas sapaan itu dengan ramah dan tanyakan bantuan. Contoh: "Halo juga! Ada yang bisa saya bantu seputar Aksara Jawi?".
    2.  **ATURAN UTAMA:** Untuk pertanyaan tentang Jawi, jawablah HANYA berdasarkan 'Konteks'.
    3.  **ATURAN FORMAT PENYAJIAN (SANGAT PENTING):**
        - Saat menjelaskan sebuah huruf, sertakan karakter Jawi-nya di dalam kurung. CONTOH: "Huruf Ca Terpisah (چ) adalah..."
        - Saat memberikan contoh kata, Anda WAJIB memberikan jawaban dalam format: TULISAN LATIN (TULISAN JAWI). CONTOH: "Tentu, contoh katanya adalah 'banyak' (باڽق)."
    4.  Jika jawabannya tidak ada di 'Konteks', katakan Anda tidak punya informasinya.

    Konteks:
    ---
    {retrieved_context}
    ---
    Pertanyaan Pengguna: {user_query}
    Jawaban Anda (ingat aturan format):
    """

    print("🧠 Mengirim prompt ke LLM...")
    try:
        headers = {"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt_template}], "max_tokens": 300, "temperature": 0.5}
        response = requests.post(QWEN_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        response_data = response.json()
        ai_response = response_data['choices'][0]['message']['content']
        print(f"💬 Jawaban dari AI diterima: '{ai_response.strip()}'")
        return jsonify({"response": ai_response.strip()})
    except Exception as e:
        print(f"❌ ERROR (Faktual): {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat-creative', methods=['POST'])
def chat_creative():
    data = request.json
    user_query = data.get('query')
    if not user_query: return jsonify({"error": "Query tidak ditemukan"}), 400
    
    print(f"\n🚀 Menerima query KREATIF: '{user_query}'")
    creative_prompt = f"""
    Anda adalah seorang guru bahasa Jawi yang kreatif.
    Tolong penuhi permintaan pengguna berikut dengan memberikan contoh atau penjelasan yang relevan dalam konteks Aksara Jawi.
    Gunakan pengetahuan Anda tentang bahasa Melayu dan Jawi untuk berkreasi.
    Permintaan Pengguna: "{user_query}"
    Jawaban Kreatif Anda:
    """
    print("🧠 Mengirim prompt KREATIF ke LLM...")
    try:
        headers = {"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": creative_prompt}], "max_tokens": 150, "temperature": 0.8}
        response = requests.post(QWEN_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        response_data = response.json()
        ai_response = response_data['choices'][0]['message']['content']
        print(f"💬 Jawaban KREATIF dari AI diterima: '{ai_response.strip()}'")
        return jsonify({"response": ai_response.strip()})
    except Exception as e:
        print(f"❌ ERROR (Kreatif): {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)