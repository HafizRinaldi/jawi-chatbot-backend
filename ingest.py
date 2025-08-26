import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

print("Memulai proses ingestion dengan data komprehensif...")

with open('jawi_knowledge.json', 'r', encoding='utf-8') as f:
    knowledge_base = json.load(f)

documents = []
for item in knowledge_base:
    if item.get('type') == 'topik_umum':
        # Proses data untuk topik umum
        text = f"Topik: {item['topik']}. Penjelasan: {item['konten']}"
        documents.append(text)
    elif item.get('type') == 'huruf':
        # Proses data untuk huruf spesifik
        text = f"Huruf: {item['nama']} ({item['bentuk']}). Posisi: {item['posisi']}. Info: {item['info']}"
        documents.append(text)

print(f"Total {len(documents)} dokumen akan diproses.")

print("Memuat model SentenceTransformer...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Membuat embeddings untuk semua dokumen...")
embeddings = model.encode(documents, convert_to_tensor=False)

print("Menyimpan index ke FAISS...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))
faiss.write_index(index, 'jawi_index.faiss')

print("Menyimpan dokumen asli...")
with open('documents.json', 'w', encoding='utf-8') as f:
    json.dump(documents, f, ensure_ascii=False, indent=4)

print("Vector database dan dokumen berhasil diperbarui dengan data komprehensif!")