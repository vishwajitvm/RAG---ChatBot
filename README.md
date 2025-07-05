# 📄 Chat with Multiple PDFs

This Streamlit application lets you **VMchat with multiple PDFs** by uploading files, extracting their text, and asking questions to get intelligent, context-aware answers.  

---

## 🚀 How it works

1️⃣ **PDF Upload**  
Upload one or more PDF files through the sidebar.  

2️⃣ **Extract Text**  
All text is extracted from the uploaded PDFs using PyPDF2.  

3️⃣ **Chunk Text**  
The extracted text is split into smaller overlapping chunks to maintain context and prepare for embedding.  

4️⃣ **Create Embeddings & Save Vector DB**  
Each chunk is converted into an embedding vector using Google Generative AI Embeddings and saved into a FAISS vector database for fast semantic search.  

5️⃣ **Ask Your Question**  
When you ask a question, it is also converted into an embedding vector.  

6️⃣ **Semantic Search**  
The app finds the most relevant chunks from your PDFs using similarity search in FAISS.  

7️⃣ **Generate Answer with LLM**  
The selected chunks and your question are passed to the Gemini language model (Google AI) to generate a detailed answer.  

8️⃣ **Display Answer & Save Conversation**  
The answer is shown in a clean, chat-style UI. All interactions are saved and can be downloaded as a CSV file.

---

## 🗺️ Workflow Summary

PDF Upload → Extract Text → Chunk Text → Create Embeddings → Save Vector DB (FAISS)
↓
User Question → Embed Question → Search Similar Chunks → Feed to LLM → Generate Answer
↓
Display Answer & Save Conversation


---

## ⚙️ Technologies Used

- **Streamlit**: For the interactive web UI.
- **PyPDF2**: To extract text from PDFs.
- **LangChain**: For text splitting, embeddings, and building QA chains.
- **Google Generative AI Embeddings**: To create semantic vectors for text and questions.
- **FAISS**: For fast vector-based similarity search.
- **Pandas**: To handle conversation history and export CSV.

---

## 💡 Author

Developed by **Vishwajit VM**  
📧 vishwaitmall50@gmail.com

---

## 🟢 Start chatting with your PDFs today!
