
# 📄 RAG Document Q&A with Groq and LLaMA3

This project is a **RAG-based (Retrieval-Augmented Generation)** Document Question Answering system built using:
- 🦙 LLaMA3 via **Groq API**
- 🤗 **Hugging Face** embeddings
- 📚 PDF ingestion and chunking
- 💬 Streamlit for interactive UI

---

## 🚀 Features

- Upload and embed research papers (PDFs)
- Ask questions related to the documents
- Retrieves the most relevant context and answers using **Groq's LLaMA3**
- Real-time document similarity insights

---

## 🧠 Tech Stack

| Component     | Tech Used                                  |
|--------------|---------------------------------------------|
| Embeddings    | `all-MiniLM-L6-v2` via Hugging Face         |
| LLM           | LLaMA3 8B via Groq API                      |
| Vector Store  | FAISS                                       |
| UI            | Streamlit                                   |
| Env Mgmt      | python-dotenv                               |

---

## 📁 Folder Structure

```

.
├── research\_papers/       # Your PDF files
├── aap.py                 # Streamlit app
├── .env                   # API keys (Groq, Hugging Face)
└── requirements.txt       # Python dependencies

````

---

## 🔐 Setup `.env`

Create a `.env` file with:

```ini
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
````

> **Make sure not to commit this file to GitHub.**

---

## 🧪 Installation & Running

1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Add your `.env` file and PDF files to `research_papers/`

4. Run the app

```bash
streamlit run aap.py
```

---

## 💡 How It Works

1. **Load & Split PDFs** into chunks
2. **Embed** chunks using `all-MiniLM-L6-v2`
3. Store in **FAISS** vector DB
4. When a question is asked:

   * Retrieve top documents
   * Generate an answer using **LLaMA3 (Groq)**

---

## ✅ Example

Upload PDFs on AI/ML, click "🔍 Create Document Embeddings", and ask:

> *"What is the core idea of the LLaMA3 paper?"*

You’ll get an accurate answer grounded in the documents!

---

## 🙏 Acknowledgments

* [LangChain](https://www.langchain.com/)
* [Groq](https://console.groq.com/)
* [Hugging Face](https://huggingface.co/)
* [Krish Naik](https://www.youtube.com/@KrishNaik)

---

## 📜 License

MIT License

```
