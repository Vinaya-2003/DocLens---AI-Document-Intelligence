# ============================================================
# APP.PY — ASKMYPDF WEB INTERFACE
# This file creates everything the user sees in the browser.
# Streamlit turns this Python script into a web app.
# Every time the user clicks something Streamlit reruns this
# entire file from top to bottom.
# Session state remembers values between those reruns.
# ============================================================

import streamlit as st
# streamlit creates the entire web interface
# Every element the user sees is created with st. commands
# st.title() = heading
# st.button() = clickable button
# st.file_uploader() = file upload area
# st.info() = blue box
# st.success() = green box
# st.error() = red box
# st.warning() = yellow box
# st.expander() = collapsible section
# st.sidebar = left side panel

import tempfile
# Creates temporary files for uploaded PDFs
# These files are automatically cleaned up after processing

import os
# Operating system tools
# Used to check if files exist and delete them

import time
# Used to track when the user was last active
# Powers the 30 minute auto session expiry feature

from rag_pipeline import (
    load_and_split_pdf,    # reads one PDF → list of chunks
    load_multiple_pdfs,    # reads many PDFs → combined chunks
    create_vector_store,   # chunks → FAISS vector store
    build_rag_chain,       # vector store → AI question answering chain
    ask_question,          # question → answer with confidence score
    generate_summary       # document → automatic summary
)
# We import from rag_pipeline.py to keep code organised
# Logic lives in rag_pipeline.py
# Interface lives in app.py

# ============================================================
# PASSWORD PROTECTION
# Simple password gate before anyone can use the app
# The password is stored in .streamlit/secrets.toml
# Streamlit Cloud reads this file securely
# ============================================================

def check_password():
    # This function shows a password input
    # Returns True if correct password entered
    # Returns False if wrong or not entered yet

    def password_entered():
        # Called when user clicks Submit
        # Checks if entered password matches our secret
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            # Delete password from memory after checking
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    # First visit — show the login form
    if "password_correct" not in st.session_state:
        st.markdown("## 🔍 DocLens")
        st.markdown("*See inside any document. Instantly.*")
        st.divider()
        st.text_input(
            "Enter password to access DocLens:",
            type="password",
            on_change=password_entered,
            key="password",
            placeholder="Enter access password"
        )
        st.caption("Contact vinayakallivalappil@gmail.com for access")
        return False

    # Wrong password entered — show error
    elif not st.session_state["password_correct"]:
        st.markdown("## 🔍 DocLens")
        st.divider()
        st.text_input(
            "Enter password to access DocLens:",
            type="password",
            on_change=password_entered,
            key="password",
            placeholder="Enter access password"
        )
        st.error("❌ Incorrect password. Please try again.")
        return False

    # Correct password — allow access
    else:
        return True

# Stop the app here if password is wrong
# Everything below this line only runs after correct password
if not check_password():
    st.stop()

# ============================================================
# PAGE CONFIGURATION
# Must be the very first Streamlit command
# Sets the browser tab title icon and page layout
# ============================================================

st.set_page_config(
    page_title="DocLens",    # shown in browser tab
    page_icon="🔍",            # icon in browser tab
    layout="wide"              # uses full width of screen
)


# ============================================================
# CUSTOM CSS STYLING
# CSS controls colours fonts spacing and visual appearance
# unsafe_allow_html=True lets us inject raw HTML and CSS
# We use this to create custom styled boxes for answers
# and confidence scores that Streamlit cannot do natively
# ============================================================

st.markdown("""
<style>
    /* Dark background for the main area */
    .main { background-color: #0f1117; }

    /* Larger bolder title text */
    .stTitle { font-size: 2.5rem !important; font-weight: 800; }

    /* Green gradient box for high confidence answers */
    .confidence-high {
        background: linear-gradient(90deg, #1a472a, #2d6a4f);
        padding: 10px 16px;
        border-radius: 8px;
        color: #95d5b2;
        font-weight: 600;
    }

    /* Orange gradient box for medium confidence answers */
    .confidence-medium {
        background: linear-gradient(90deg, #3d2b00, #7b4f00);
        padding: 10px 16px;
        border-radius: 8px;
        color: #ffd166;
        font-weight: 600;
    }

    /* Red gradient box for low confidence answers */
    .confidence-low {
        background: linear-gradient(90deg, #3b0a0a, #6b1a1a);
        padding: 10px 16px;
        border-radius: 8px;
        color: #ff6b6b;
        font-weight: 600;
    }

    /* Purple left border box for displaying answers */
    .answer-box {
        background: #1e1e2e;
        border-left: 4px solid #7c3aed;
        padding: 16px;
        border-radius: 8px;
        color: #e2e8f0;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALISATION
# Streamlit reruns the entire script on every user action.
# Session state persists values between those reruns.
# We check if each key exists before setting it so we do not
# accidentally reset values when the script reruns.
# ============================================================

# rag_chain holds the AI pipeline — None means not ready
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# chat_history holds all Q&A pairs as a list of dictionaries
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# doc_names holds the original names of loaded documents
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []

# summary holds the auto generated document summary text
if "summary" not in st.session_state:
    st.session_state.summary = None

# last_active holds the timestamp of the last user action
if "last_active" not in st.session_state:
    st.session_state.last_active = time.time()

# prefill holds text to pre-fill the question input box
# Set when user clicks a suggestion button
if "prefill" not in st.session_state:
    st.session_state.prefill = ""

# total_chunks holds how many sections were indexed
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0


# ============================================================
# AUTO SESSION EXPIRY — DATA PROTECTION
# If user is inactive for 30 minutes clear all data.
# 1800 seconds = 30 minutes.
# This ensures no user data lingers on the server.
# ============================================================

# Only check expiry if a document is actually loaded
if st.session_state.rag_chain is not None:

    # time.time() returns current time as seconds since 1970
    # Subtracting last_active gives seconds of inactivity
    time_inactive = time.time() - st.session_state.last_active

    if time_inactive > 1800:
        # Clear everything from memory
        st.session_state.rag_chain = None
        st.session_state.chat_history = []
        st.session_state.doc_names = []
        st.session_state.summary = None
        st.session_state.total_chunks = 0
        st.warning(
            "⏰ Session expired after 30 minutes. "
            "All data cleared. Please upload again."
        )

# Update last active timestamp on every page load
st.session_state.last_active = time.time()


# ============================================================
# SIDEBAR — LEFT PANEL
# Contains session info privacy policy and controls
# with st.sidebar: puts everything inside the left panel
# ============================================================

with st.sidebar:

    st.markdown("## 🔍 DocLens")
    st.markdown("*See inside any document. Instantly.*")
    st.divider()

    # Show document info only when something is loaded
    if st.session_state.doc_names:

        st.markdown("### 📂 Loaded Documents")

        # Loop through each document name and display it
        for name in st.session_state.doc_names:
            st.markdown("✅ " + name)

        st.divider()

        # Show stats in two side by side columns
        # st.columns(2) creates two equal width columns
        col1, col2 = st.columns(2)
        with col1:
            # st.metric shows a number in a highlighted card
            st.metric("Questions", len(st.session_state.chat_history))
        with col2:
            st.metric("Sections", st.session_state.total_chunks)

        st.divider()

        # Download button — only show if there are answers to download
        if st.session_state.chat_history:

            # Build the content of the download file as a string
            download_text = "ASKMYPDF — Q&A HISTORY\n"
            download_text += "=" * 40 + "\n\n"

            # Add document names to the file header
            for name in st.session_state.doc_names:
                download_text += "Document: " + name + "\n"

            download_text += "\n" + "=" * 40 + "\n\n"

            # Add each question answer and confidence score
            for i, item in enumerate(st.session_state.chat_history, 1):
                download_text += "Q" + str(i) + ": " + item["question"] + "\n"
                download_text += "Answer: " + item["answer"] + "\n"
                download_text += (
                    "Confidence: " + item["confidence_label"] +
                    " (" + str(item["confidence_pct"]) + "%)\n"
                )
                download_text += "-" * 40 + "\n\n"

            # st.download_button creates a clickable link
            # When clicked the browser downloads a .txt file
            # data = the content of the file
            # file_name = what the downloaded file is called
            # mime = the file type (text/plain = .txt)
            st.download_button(
                label="📥 Download Q&A History",
                data=download_text,
                file_name="doclens_history.txt",
                mime="text/plain",
                use_container_width=True
            )

        st.divider()

        # Clear session button
        # Wipes all data from memory immediately
        if st.button("🗑️ Clear Session", use_container_width=True):
            st.session_state.rag_chain = None
            st.session_state.chat_history = []
            st.session_state.doc_names = []
            st.session_state.summary = None
            st.session_state.total_chunks = 0
            # st.rerun() restarts the app immediately
            # This refreshes the interface to show empty state
            st.rerun()

    else:
        # Show when no document is loaded
        st.info("Upload PDFs to get started")

    st.divider()

    # Privacy policy in a collapsible section
    # st.expander() creates a section the user can open and close
    with st.expander("🔒 Privacy Policy"):
        st.markdown("""
- PDF deleted immediately after processing
- Nothing stored permanently on any server
- No data shared with anyone
- Auto clears after 30 minutes of inactivity
        """)

    # Supported document types in a collapsible section
    with st.expander("📂 Supported Documents"):
        st.markdown("""
- Financial and annual reports
- Legal documents and contracts
- Research and academic papers
- HR policies and handbooks
- Medical and clinical reports
- Government policy documents
- Product manuals and guides
- Any other text based PDF
        """)


# ============================================================
# MAIN PAGE HEADER
# ============================================================

st.title("🔍 DocLens")
st.markdown(
    "**See inside any document. Instantly.** Upload any PDF and ask questions in plain English."
)

# Privacy notice in a prominent green box at the top
st.success(
    "🔒 Privacy First — Your documents are deleted immediately "
    "after processing. Nothing is stored permanently."
)

st.divider()


# ============================================================
# FILE UPLOAD SECTION
# accept_multiple_files=True allows selecting more than one PDF
# This fills the single document gap of tools like ChatPDF
# ============================================================

st.subheader("📤 Upload Documents")
st.caption(
    "Upload one or multiple PDFs. "
    "Ask questions across all documents at once."
)

# File uploader widget
# type=["pdf"] = only PDF files allowed
# accept_multiple_files=True = can select multiple files
# help = tooltip shown when user hovers over the question mark
uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Maximum 10MB per file"
)

# Only run this block if at least one file was uploaded
if uploaded_files:

    # Validate each file is under the 10MB size limit
    # Large files can cause slow processing or crashes
    all_valid = True

    for f in uploaded_files:
        # Convert bytes to megabytes (1MB = 1024 * 1024 bytes)
        size_mb = f.size / (1024 * 1024)

        if size_mb > 10:
            st.error(
                "❌ " + f.name + " exceeds 10MB limit (" +
                str(round(size_mb, 1)) + "MB)"
            )
            all_valid = False

    # Only show processing option if all files pass size check
    if all_valid:

        # Show each file with its size in a green success box
        for f in uploaded_files:
            size_mb = round(f.size / (1024 * 1024), 2)
            st.success("✅ " + f.name + " (" + str(size_mb) + " MB)")

        # Process button — triggers the entire pipeline
        # use_container_width=True makes button full width
        if st.button(
            "🚀 Process Documents",
            type="primary",
            use_container_width=True
        ):

            # st.empty() creates an updateable placeholder
            # We update it with different messages as we progress
            # This gives users real time feedback during processing
            progress = st.empty()

            # tmp_paths stores paths of temporary PDF files on disk
            # Defined BEFORE try block so finally can always access it
            tmp_paths = []

            # tmp_names stores the original file names
            # Used to label which document each answer came from
            tmp_names = []

            # try = run the main processing code
            # except = if anything fails show a friendly error
            # finally = always runs — deletes temp files no matter what
            try:

                # STEP 1 — Save uploaded files to disk temporarily
                # PyPDFLoader needs files on disk not in memory
                progress.info("🔒 Receiving documents securely...")

                for f in uploaded_files:
                    # NamedTemporaryFile creates a file with random name
                    # delete=False = we control when to delete it
                    # suffix=".pdf" = gives it the correct extension
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as tmp:
                        # Write the uploaded file bytes to temp file
                        tmp.write(f.getvalue())
                        # Save the temp path for loading and deletion
                        tmp_paths.append(tmp.name)
                        # Save original name for labelling
                        tmp_names.append(f.name)

                # STEP 2 — Load and split PDFs into chunks
                if len(tmp_paths) == 1:
                    # Single PDF — use simple loader
                    progress.info("📖 Reading document...")
                    chunks = load_and_split_pdf(tmp_paths[0])

                    # Label each chunk with the document name
                    for chunk in chunks:
                        chunk.metadata["source_document"] = tmp_names[0]

                else:
                    # Multiple PDFs — use combined loader
                    # load_multiple_pdfs handles labelling internally
                    progress.info(
                        "📖 Reading " + str(len(tmp_paths)) +
                        " documents..."
                    )
                    chunks = load_multiple_pdfs(tmp_paths, tmp_names)

                # STEP 3 — Convert chunks to vectors and store in FAISS
                progress.info(
                    "🧠 Building knowledge index from " +
                    str(len(chunks)) + " sections..."
                )
                # create_vector_store now returns only the vector store
                vector_store = create_vector_store(chunks)

                # STEP 4 — Build the RAG question answering chain
                progress.info("⚡ Preparing AI assistant...")
                rag_chain = build_rag_chain(vector_store)

                # STEP 5 — Generate automatic summary
                # Runs immediately so users understand the document
                # before asking any questions
                progress.info("📋 Generating summary...")
                summary = generate_summary(rag_chain)

                # STEP 6 — Save everything to session state
                # This makes data available for the rest of the session
                st.session_state.rag_chain = rag_chain
                st.session_state.chat_history = []
                st.session_state.doc_names = tmp_names
                st.session_state.summary = summary
                st.session_state.total_chunks = len(chunks)

                # Show final success message
                progress.success(
                    "✅ " + str(len(tmp_names)) +
                    " document(s) ready. " +
                    str(len(chunks)) + " sections indexed."
                )

            except Exception as e:
                # Exception catches any error that occurred
                # str(e) converts the technical error to readable text
                progress.error(
                    "❌ Error: " + str(e) + " Please try again."
                )

            finally:
                # This block ALWAYS runs — success or failure
                # Guarantees temp PDFs are always deleted from server
                for tmp_path in tmp_paths:
                    try:
                        if os.path.exists(tmp_path):
                            # os.unlink permanently deletes the file
                            os.unlink(tmp_path)
                    except:
                        # pass = do nothing if deletion fails
                        # Prevents a deletion error from crashing the app
                        pass


# ============================================================
# AUTO SUMMARY DISPLAY
# Shows immediately after document is processed
# Users understand the document without asking any questions
# This fills a gap in all existing free PDF chat tools
# ============================================================

if st.session_state.summary:
    st.divider()
    st.subheader("📋 Document Summary")
    # st.info displays content in a blue highlighted box
    st.info(st.session_state.summary)


# ============================================================
# QUESTION AND ANSWER SECTION
# Only shows when a document is loaded
# ============================================================

if st.session_state.rag_chain is not None:

    st.divider()
    st.subheader("💬 Ask a Question")

    # Show tip when multiple documents are loaded
    if len(st.session_state.doc_names) > 1:
        st.caption(
            "Ask about any single document or compare across all."
        )

    # Suggestion buttons — help users know what to ask
    # Four columns creates four buttons side by side
    col1, col2, col3, col4 = st.columns(4)

    # When clicked each button sets the prefill text
    # which fills the question input automatically
    with col1:
        if st.button("📋 Key Points", use_container_width=True):
            st.session_state.prefill = (
                "What are the key points in this document?"
            )
    with col2:
        if st.button("🔢 Key Numbers", use_container_width=True):
            st.session_state.prefill = (
                "What are the important numbers and statistics?"
            )
    with col3:
        if st.button("✅ Conclusions", use_container_width=True):
            st.session_state.prefill = (
                "What are the main conclusions or recommendations?"
            )
    with col4:
        if st.button("🔄 Compare", use_container_width=True):
            st.session_state.prefill = (
                "What are the main differences between the documents?"
            )

    # Text input for the question
    # value= pre-fills with suggestion text if button was clicked
    question = st.text_input(
        "Your question:",
        value=st.session_state.prefill,
        placeholder="e.g. What is the main purpose of this document?"
    )

    # Reset prefill after it has been used
    # Prevents the box from staying filled on every rerun
    if st.session_state.prefill:
        st.session_state.prefill = ""

    # Get Answer button
    # question.strip() removes spaces to prevent empty submissions
    if st.button(
        "🔍 Get Answer",
        type="primary",
        use_container_width=True
    ) and question.strip():

        with st.spinner("Searching and generating answer..."):
            try:
                # Call ask_question from rag_pipeline.py
                # Returns answer confidence score and source info
                result = ask_question(
                    st.session_state.rag_chain,
                    question
                )
                # Add to history so it appears in the answers section
                st.session_state.chat_history.append(result)

            except Exception as e:
                st.error("❌ Could not generate answer: " + str(e))


    # ============================================================
    # DISPLAY ANSWERS WITH CONFIDENCE SCORES
    # reversed() shows most recent answer at the top
    # enumerate() gives us both the index and the item
    # ============================================================

    if st.session_state.chat_history:

        st.divider()
        st.subheader("📝 Answers")
        st.caption(
            str(len(st.session_state.chat_history)) +
            " question(s) asked this session"
        )

        # Loop through answers newest first
        for i, item in enumerate(
            reversed(st.session_state.chat_history)
        ):
            # Calculate question number (newest = highest number)
            q_num = len(st.session_state.chat_history) - i

            # Display question in bold text
            st.markdown(
                "**Q" + str(q_num) + ": " + item["question"] + "**"
            )

            # Display answer in custom styled box
            # unsafe_allow_html=True allows our CSS class to work
            st.markdown(
                '<div class="answer-box">💡 ' +
                item["answer"] + '</div>',
                unsafe_allow_html=True
            )

            # Display confidence score with colour coded box
            # High = green, Medium = orange, Low = red
            # This is the key feature no free tool provides
            conf = item["confidence_label"]
            pct = str(item["confidence_pct"])

            if conf == "High":
                st.markdown(
                    '<div class="confidence-high">✅ Confidence: ' +
                    conf + ' (' + pct + '%) — ' +
                    'Document clearly contains this information</div>',
                    unsafe_allow_html=True
                )
            elif conf == "Medium":
                st.markdown(
                    '<div class="confidence-medium">⚠️ Confidence: ' +
                    conf + ' (' + pct + '%) — ' +
                    'Document partially covers this topic</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="confidence-low">❌ Confidence: ' +
                    conf + ' (' + pct + '%) — ' +
                    'Information may not be in document</div>',
                    unsafe_allow_html=True
                )

            # Show which document the answer came from
            # Especially useful with multiple documents loaded
            if item.get("source_documents"):
                st.caption(
                    "📄 Source: " +
                    ", ".join(item["source_documents"])
                )

            # Collapsible section showing exact document chunks used
            # Users can verify every answer against the original text
            with st.expander(
                "📚 View source sections (" +
                str(len(item["source_chunks"])) + " sections used)"
            ):
                st.caption(
                    "These are the exact parts of your document "
                    "the AI read to generate this answer."
                )

                for j, chunk in enumerate(item["source_chunks"], 1):
                    st.markdown("**Section " + str(j) + ":**")

                    # Show first 500 characters with ... if longer
                    if len(chunk) > 500:
                        display = chunk[:500] + "..."
                    else:
                        display = chunk

                    # st.text shows fixed width formatted text
                    st.text(display)

                    # Add divider between chunks not after last one
                    if j < len(item["source_chunks"]):
                        st.markdown("---")

            st.divider()

else:
    # Show when no document is loaded yet
    if not uploaded_files:
        st.info(
            "👆 Upload a PDF above to get started."
        )


# ============================================================
# FOOTER — BOTTOM OF PAGE
# ============================================================

st.markdown("---")

# Three equal columns for footer content
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🔒 Data never stored permanently")
with col2:
    st.caption("⚡ LangChain · FAISS · HuggingFace · Groq")
with col3:
    st.caption("Built by Vinaya K")