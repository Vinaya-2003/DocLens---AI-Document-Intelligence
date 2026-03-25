# ============================================================
# RAG PIPELINE — THE BRAIN OF ASKMYPDF
# This file handles all the AI logic:
# 1. Reading PDFs and splitting into chunks
# 2. Converting chunks to vectors (numbers)
# 3. Storing vectors in FAISS for fast search
# 4. Building the AI chain that answers questions
# 5. Calculating confidence scores
# 6. Generating automatic document summaries
# ============================================================

import os
# os = operating system tools
# Used to read the GROQ_API_KEY from the .env file

from dotenv import load_dotenv
# dotenv loads our secret keys from .env file
# Keeps API keys out of the code

from langchain_community.document_loaders import PyPDFLoader
# PyPDFLoader opens a PDF file and reads every page
# Returns a list of Document objects with text and metadata

from langchain_text_splitters import RecursiveCharacterTextSplitter
# Cuts long text into smaller overlapping chunks
# chunk_size controls how big each piece is
# chunk_overlap ensures no information is lost at boundaries

from langchain_huggingface import HuggingFaceEmbeddings
# Converts text into vectors — lists of numbers
# Similar sentences get similar numbers
# This is how the app understands meaning not just keywords

from langchain_community.vectorstores import FAISS
# FAISS stores all our vectors and searches them at high speed
# Built by Facebook — can search millions of vectors instantly

from langchain_groq import ChatGroq
# Connects to Groq's free LLaMA AI model
# This is what reads the chunks and writes the answer

from langchain_core.prompts import PromptTemplate
# A template that tells the AI exactly how to behave
# We insert the context chunks and question into this template

from langchain_core.runnables import RunnablePassthrough
# Passes the question unchanged through the chain
# Think of it as a pipe that lets data flow through

from langchain_core.output_parsers import StrOutputParser
# Converts the AI model output into a clean readable string
# Without this the output is a complex object not plain text

# Load secret keys from .env file into environment
load_dotenv()


# ============================================================
# FUNCTION 1 — LOAD AND SPLIT ONE PDF
# ============================================================

def load_and_split_pdf(file_path):
    # file_path is the location of the PDF on the computer
    # Example: C:\Temp\tmpXXXXXX.pdf

    # PyPDFLoader reads every page of the PDF
    # Each page becomes a Document object with:
    # .page_content = the text on that page
    # .metadata = page number and source filename
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # RecursiveCharacterTextSplitter cuts text into chunks
    # chunk_size=1000 means each chunk is ~1000 characters
    # chunk_overlap=200 means chunks share 200 characters
    # with their neighbour to avoid losing context at edges
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    # split_documents takes pages and returns smaller chunks
    chunks = splitter.split_documents(pages)
    return chunks


# ============================================================
# FUNCTION 2 — LOAD AND SPLIT MULTIPLE PDFs
# NEW FEATURE — fills gap of single document limitation
# in tools like ChatPDF which only support one PDF at a time
# ============================================================

def load_multiple_pdfs(file_paths, file_names):
    # file_paths is a list of temporary file locations
    # file_names is a list of original uploaded file names
    # We need file_names to label which document each chunk came from

    all_chunks = []
    # all_chunks will collect chunks from ALL documents combined

    # Loop through each PDF file
    for i, file_path in enumerate(file_paths):
        # i is the index number (0, 1, 2...)
        # file_path is the temp location of that PDF

        # Load and split this individual PDF
        chunks = load_and_split_pdf(file_path)

        # Add the original filename to each chunk's metadata
        # This tells us WHICH document each answer came from
        # metadata is like a label attached to each chunk
        for chunk in chunks:
            chunk.metadata["source_document"] = file_names[i]

        # Add this document's chunks to the combined list
        all_chunks.extend(chunks)

    return all_chunks


# ============================================================
# FUNCTION 3 — CREATE VECTOR STORE
# ============================================================

def create_vector_store(chunks):
    # HuggingFaceEmbeddings loads a pre-trained model that
    # converts any text into a vector of 384 numbers
    # all-MiniLM-L6-v2 is small fast and works well for search
    # device=cpu means it runs on the normal processor
    # normalize_embeddings=True makes vectors easier to compare
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # FAISS.from_documents does two things at once:
    # 1. Converts every chunk to a vector using embedding_model
    # 2. Stores all vectors in a FAISS index for fast searching
    vector_store = FAISS.from_documents(chunks, embedding_model)

    # Return only the vector store
    # We no longer need the embedding model after this
    return vector_store


# ============================================================
# HELPER — FORMAT DOCUMENT CHUNKS INTO ONE TEXT BLOCK
# ============================================================

def format_docs(docs):
    # docs is a list of Document objects
    # The prompt template needs one single block of text
    # This function joins all chunks with blank lines between them
    # "\n\n".join() puts two newlines between each chunk
    return "\n\n".join(doc.page_content for doc in docs)


# ============================================================
# FUNCTION 4 — BUILD THE RAG CHAIN
# ============================================================

def build_rag_chain(vector_store):
    # RAG = Retrieval Augmented Generation
    # Retrieval = find relevant chunks from the document
    # Augmented = add those chunks to the AI prompt
    # Generation = AI writes an answer using only those chunks
    #
    # This prevents the AI from making things up because
    # it can only use what we give it in the prompt

    # ChatGroq connects to Groq's free LLaMA 3.3 70B model
    # temperature=0 = focused factual answers not creative ones
    # Higher temperature = more creative but less accurate
    # max_tokens=1024 = maximum answer length (~750 words)
    # api_key reads our secret key from the .env file
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # The retriever searches FAISS for relevant chunks
    # search_type="similarity" finds chunks most similar to question
    # k=4 returns the 4 most relevant chunks
    # 4 chunks gives enough context without overwhelming the AI
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # The prompt template is the exact instruction sent to the AI
    # {context} gets replaced with the 4 retrieved chunks
    # {question} gets replaced with the user's question
    # We tell the AI NOT to make things up — this prevents
    # hallucination which is when AI invents false information
    prompt = PromptTemplate.from_template("""
You are a helpful and accurate document assistant.
Answer the question using ONLY the information from the context below.
If the answer is not in the context say:
"I could not find this information in the document."
Do NOT guess or make up any information.
Keep your answer clear and easy to understand.

Context:
{context}

Question: {question}

Answer:""")

    # The chain connects all pieces like an assembly line
    # | means pipe — output of one step goes into the next
    #
    # Step by step:
    # 1. Question comes in
    # 2. retriever finds 4 most similar chunks from FAISS
    # 3. format_docs joins those chunks into one text block
    # 4. RunnablePassthrough passes the question unchanged
    # 5. prompt combines chunks + question into full instruction
    # 6. llm reads the instruction and generates an answer
    # 7. StrOutputParser converts output to clean string
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Return both chain and retriever
    # chain is used to generate answers
    # retriever is used separately to get source chunks + scores
    return chain, retriever


# ============================================================
# FUNCTION 5 — ASK A QUESTION WITH CONFIDENCE SCORE
# NEW FEATURE — no other free PDF tool provides this
# ============================================================

def ask_question(rag_chain_tuple, question):
    # Unpack the tuple we got from build_rag_chain
    chain, retriever = rag_chain_tuple

    # Run the full chain to get the answer
    # chain.invoke() triggers the entire pipeline end to end
    answer = chain.invoke(question)

    # Get the FAISS vector store from the retriever
    # We need it to get similarity scores
    vector_store = retriever.vectorstore

    # similarity_search_with_score returns pairs of (document, score)
    # The score is a distance value — lower means MORE similar
    # Distance 0 = exact match, Distance 2 = completely different
    docs_with_scores = vector_store.similarity_search_with_score(
        question, k=4
    )

    # Extract just the scores from the pairs
    scores = [score for _, score in docs_with_scores]

    # Calculate the average score across all 4 retrieved chunks
    avg_score = sum(scores) / len(scores) if scores else 1.0

    # Convert FAISS distance to a 0 to 100 confidence percentage
    # Formula: confidence = (1 - distance/2) * 100
    # Distance 0 → confidence 100%
    # Distance 1 → confidence 50%
    # Distance 2 → confidence 0%
    # max(0, min(100, ...)) clamps the value between 0 and 100
    confidence_pct = max(0, min(100, round((1 - avg_score / 2) * 100)))

    # Assign a human readable label based on the percentage
    if confidence_pct >= 70:
        confidence_label = "High"
    elif confidence_pct >= 40:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"

    # Get the text content from each retrieved chunk
    source_chunks = [doc.page_content for doc, _ in docs_with_scores]

    # Get the unique document names from chunk metadata
    # set() removes duplicates if multiple chunks came from same doc
    # list() converts the set back to a list
    source_names = list(set([
        doc.metadata.get("source_document", "Document")
        for doc, _ in docs_with_scores
    ]))

    # Return everything as a dictionary
    # The app.py file uses these values to display the answer
    return {
        "question": question,
        "answer": answer,
        "confidence_pct": confidence_pct,
        "confidence_label": confidence_label,
        "source_chunks": source_chunks,
        "source_documents": source_names
    }


# ============================================================
# FUNCTION 6 — GENERATE AUTOMATIC DOCUMENT SUMMARY
# NEW FEATURE — shows what the document is about immediately
# Most PDF chat tools require you to ask questions first
# This gives users instant context without any effort
# ============================================================

def generate_summary(rag_chain_tuple):
    # We simply ask the RAG chain a structured summary question
    # The chain finds the most relevant chunks across the whole
    # document and generates a structured response
    question = """
    Please provide a structured summary of this document with:
    1. What this document is about in 2 to 3 sentences
    2. The 5 most important points or findings
    3. Any key numbers dates or names mentioned
    Keep it clear and concise.
    """
    result = ask_question(rag_chain_tuple, question)
    # Return just the answer text not the full result dictionary
    return result["answer"]