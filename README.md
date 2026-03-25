# Live Demo
doclens---ai-document-intelligence-gr2f26cvy9fkxd45njxn4a.streamlit.app

# 🔍 DocLens — See Inside Any Document. Instantly.

> An AI powered document intelligence web application that lets you upload any PDF and ask questions about it in plain English. No reading required.

---

## What is DocLens?

DocLens is a RAG (Retrieval Augmented Generation) powered web application that answers questions from any PDF document accurately and instantly. Instead of reading through hundreds of pages manually, upload your document and ask questions in plain English.

Works with any PDF — financial reports, legal documents, research papers, HR policies, medical reports, government documents, product manuals and more.

---

## Key Features

| Feature | Description |
|---------|-------------|
| Multi PDF Support | Upload and query multiple PDFs simultaneously |
| Confidence Scoring | Every answer comes with a reliability percentage |
| Auto Summary | Document summarised automatically on upload |
| Source Transparency | See exactly which document section each answer came from |
| Download History | Export full Q&A session as a text file |
| Privacy First | PDF deleted immediately after processing |
| Session Isolation | Each user's data completely separate |
| Auto Expiry | All data cleared after 30 minutes of inactivity |

---

## What Makes DocLens Different

| Feature | DocLens | ChatPDF | AskYourPDF |
|---------|---------|---------|------------|
| Multi PDF support | Yes | No | No |
| Confidence scoring | Yes | No | No |
| Auto document summary | Yes | No | No |
| Source transparency | Full | Partial | Partial |
| Download Q&A history | Yes | Paid only | Paid only |
| Daily usage limits | None | 2 PDFs/day | Limited |
| Cost | Free | Freemium | Freemium |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| Web Interface | Streamlit |
| AI Orchestration | LangChain |
| Language Model | Groq LLaMA 3.3 70B (free) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (free) |
| Vector Database | FAISS by Facebook AI |
| PDF Reading | PyPDF |

---

## How It Works

```
PDF uploaded
     ↓
PyPDF reads all pages
     ↓
Text split into 1000 character chunks
     ↓
HuggingFace converts chunks to vectors
     ↓
FAISS stores vectors in memory
     ↓
PDF deleted from server immediately
     ↓
User asks a question
     ↓
FAISS finds 4 most relevant chunks
     ↓
Chunks + question sent to Groq LLaMA
     ↓
Answer + confidence score + source displayed
```

---

## How to Run Locally

```bash
git clone https://github.com/your-username/doclens.git
cd doclens
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```
Get your free key at console.groq.com

```bash
streamlit run app.py
```

---

## Project Structure

```
doclens/
├── app.py                     # Web interface
├── rag_pipeline.py            # AI pipeline
├── requirements.txt           # Dependencies
├── .env                       # API keys (never uploaded)
├── .gitignore                 # Excluded files
├── .streamlit/secrets.toml   # Streamlit Cloud secrets
├── LICENSE                    # Copyright
├── README.md                  # This file
└── PROJECT_DOCUMENTATION.md  # Detailed explanation
```

---

## Author

**Vinaya K** — vinayakallivalappil@gmail.com — Palakkad, Kerala

---

## License

Copyright 2026 Vinaya K. All rights reserved.
