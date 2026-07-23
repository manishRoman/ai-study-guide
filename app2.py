import os
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
import streamlit as st
from pydantic import BaseModel
import json

from google.genai import types

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ==========================================
# 1. Configuration, Setup & Schemas
# ==========================================
st.set_page_config(page_title="My PDF Chatbot", page_icon="🤖", layout="wide")

GEMINI_API_KEY = (os.environ.get("GEMINI_API_KEY", "").strip() or st.secrets.get("gemini_api_key"))
if not GEMINI_API_KEY:
    st.error("Missing GEMINI_API_KEY. Set it in .env, environment variables, or Streamlit secrets.")
    st.stop()

# --- Pydantic Schemas for Structured JSON Flashcards ---
class Flashcard(BaseModel):
    front: str
    back: str

class FlashcardDeck(BaseModel):
    cards: list[Flashcard]

# We use @st.cache_resource so the database doesn't reload every time you click a button
@st.cache_resource
def init_services():
    client = genai.Client(api_key=GEMINI_API_KEY)
    chroma_client = chromadb.PersistentClient(path="./my_vector_db")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = chroma_client.get_or_create_collection(
        name="my_pdf_collection", 
        embedding_function=sentence_transformer_ef
    )
    return client, collection

client, collection = init_services()


# ==========================================
# 2. Phase 1: Ingestion (Upgraded for Citations)
# ==========================================
def process_pdf(uploaded_file):
    # 1. Save the uploaded file temporarily
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # 2. Open the PDF
        doc = fitz.open(temp_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        all_chunks = []
        all_metadatas = []
        all_ids = []

        # 3. Process page by page instead of all at once
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            if not page_text.strip():
                continue
            
            # Split the text of just THIS specific page
            page_chunks = text_splitter.split_text(page_text)
            
            for i, chunk in enumerate(page_chunks):
                all_chunks.append(chunk)
                
                # Tag this chunk with the filename and page number
                all_metadatas.append({
                    "source": uploaded_file.name, 
                    "page": page_num + 1  # Humans start counting pages at 1
                })
                all_ids.append(f"{uploaded_file.name}_p{page_num+1}_c{i}")
        
        # 4. Close the file so Windows doesn't get angry (PermissionError fix)
        doc.close() 

        # 5. Add everything to the Vector DB with Metadata
        if all_chunks:
            collection.add(
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )
            st.success(f"Successfully memorized {len(all_chunks)} chunks from {uploaded_file.name}!")
            
    finally:
        # 6. Cleanup: Delete the temp file from your hard drive
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ==========================================
# 3. Flashcard Generation Logic
# ==========================================
def generate_flashcards(context: str, num_cards: int = 5):
    """Generates structured flashcards from text using Gemini."""
    try:
        prompt = f"""
        Analyze the following text context and generate exactly {num_cards} high-quality study flashcards.
        The 'front' should contain a precise question, concept, or term from the text.
        The 'back' should contain a clear, concise explanation or answer.
        
        Context:
        {context}
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FlashcardDeck,
                temperature=0.3
            ),
        )
        return json.loads(response.text)["cards"]
    except Exception as e:
        st.error(f"Failed to generate flashcards: {e}")
        return []


# ==========================================
# 4. Session State Initialization
# ==========================================
# Chatbot History State
if "messages" not in st.session_state:
    st.session_state.messages = []

# New Flashcard Interaction States
if "flashcards" not in st.session_state:
    st.session_state.flashcards = []
if "card_index" not in st.session_state:
    st.session_state.card_index = 0
if "is_flipped" not in st.session_state:
    st.session_state.is_flipped = False
if "mastered_count" not in st.session_state:
    st.session_state.mastered_count = 0


# ==========================================
# 5. Streamlit UI Layout
# ==========================================
st.title("📄StudyRAG")

# --- Sidebar: File Uploading, Settings & Navigation ---
with st.sidebar:
    st.header("1. Document Management")
    
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process All Documents"):
            with st.spinner("Reading and memorizing documents..."):
                for uploaded_file in uploaded_files:
                    process_pdf(uploaded_file)
            st.success("Documents added to memory!")

    st.divider()
    
    st.header("2. Search & Filter Settings")
    existing_metadatas = collection.get(include=["metadatas"])['metadatas']
    all_sources = list(set([m['source'] for m in existing_metadatas if m and 'source' in m]))
    
    selected_docs = st.multiselect(
        "Select documents to use (leave empty for everything):",
        options=all_sources,
        default=[]
    )
    
    st.divider()
    
    # --- NEW: App Navigation in Sidebar ---
    st.header("3. App Navigation")
    app_mode = st.radio("Choose a mode:", ["💬 AI Chatbot Q&A", "🎓 Flashcard Study Deck"])


# ------------------------------------------
# VIEW 1: AI Chatbot (Snaps input to bottom)
# ------------------------------------------
if app_mode == "💬 AI Chatbot Q&A":
    st.header("Chat with your Data")

    # 1. Display all chat messages and the Custom Flashcard forms
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Attach a flashcard builder to every AI answer
            if message["role"] == "assistant":
                with st.expander("➕ Create Custom Flashcard from this answer"):
                    with st.form(key=f"flashcard_form_{i}"):
                        user_front = st.text_input("Card Front (Question/Topic):", placeholder="e.g., What is KNN?")
                        user_back = st.text_area("Card Back (Answer/Explanation):", value=message["content"], height=100)
                        
                        if st.form_submit_button("💾 Save to Study Deck"):
                            if user_front.strip() and user_back.strip():
                                st.session_state.flashcards.append({
                                    "front": user_front, 
                                    "back": user_back
                                })
                                st.success("Saved! Switch to the Flashcard mode to study it.")
                            else:
                                st.error("Please fill out both the Front and Back.")

    # 2. React to user typing a question (This will now stick to the bottom)
    if prompt := st.chat_input("Ask a question about the PDF..."):
        
        # Build Chat History
        history_context = ""
        recent_history = st.session_state.messages[-5:] 
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_context += f"{role}: {msg['content']}\n"

        # Build Database Filter
        if selected_docs:
            if len(selected_docs) == 1:
                where_filter = {"source": selected_docs[0]}
            else:
                where_filter = {"source": {"$in": selected_docs}}
        else:
            where_filter = None

        # Retrieve Context
        if where_filter:
            results = collection.query(query_texts=[prompt], n_results=3, where=where_filter)
        else:
            results = collection.query(query_texts=[prompt], n_results=3)
        
        # Format Response
        if not results['documents'][0]:
            response_text = "I couldn't find any relevant information in the document."
        else:
            context_list = []
            for text, meta in zip(results['documents'][0], results['metadatas'][0]):
                citation = f"[Source: {meta['source']}, Page: {meta['page']}]"
                context_list.append(f"{citation}\nContent: {text}")
                
            final_context = "\n\n".join(context_list)   

            gemini_prompt = f"""
            You are a helpful document assistant. Use the context and history provided below to answer the user's question.

            --- CHAT HISTORY ---
            {history_context}

            --- RETRIEVED DOCUMENT CONTEXT ---
            {final_context}

            --- NEW QUESTION ---
            {prompt}

            STYLE RULES:
            1. Use the Chat History to understand context/pronouns (like "it" or "that").
            2. Write a clean, natural response without parenthetical citations in the middle of sentences.
            3. At the end of your response, add a thin horizontal line and a section called '📌 SOURCES'.
            4. In the 'SOURCES' section, list the unique filenames and page numbers you actually used.
            """

            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=gemini_prompt,
            )
            response_text = response.text
        
        # 3. Save both messages to memory IMMEDIATELY
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        
        # 4. Force Streamlit to redraw the page to show new message & expander
        st.rerun()


# ------------------------------------------
# VIEW 2: The Flashcard UI Stage
# ------------------------------------------
elif app_mode == "🎓 Flashcard Study Deck":
    st.header("Active Learning Mode")
    st.write("Generate custom study questions directly from your selected documents, or review cards you saved from the chat.")

    if st.button("✨ Auto-Generate Flashcards from Document Memory", type="primary"):
        # Pull existing chunks from Chroma based on selected documents
        if selected_docs:
            if len(selected_docs) == 1:
                db_data = collection.get(where={"source": selected_docs[0]}, limit=15)
            else:
                db_data = collection.get(where={"source": {"$in": selected_docs}}, limit=15)
        else:
            db_data = collection.get(limit=15) # Grab standard sample if nothing filtered

        # Extract text strings from DB payload
        if db_data and db_data['documents']:
            combined_document_text = "\n".join(db_data['documents'])
            
            with st.spinner("Analyzing DB segments and structuring study deck..."):
                st.session_state.flashcards = generate_flashcards(combined_document_text, num_cards=5)
                st.session_state.card_index = 0
                st.session_state.is_flipped = False
                st.session_state.mastered_count = 0
                st.rerun()
        else:
            st.warning("Your vector database is empty. Please upload and process a PDF file first!")

    # Flashcard Viewer Render Loop
    if st.session_state.flashcards:
        cards = st.session_state.flashcards
        
        # Safety check in case custom cards were added but index is out of bounds
        if st.session_state.card_index >= len(cards):
            st.session_state.card_index = 0
            
        idx = st.session_state.card_index
        current_card = cards[idx]
        
        # Performance Tracking Layout
        col_metrics1, col_metrics2 = st.columns(2)
        with col_metrics1:
            st.metric(label="Progress", value=f"Card {idx + 1} of {len(cards)}")
        with col_metrics2:
            st.metric(label="Mastered Cards", value=st.session_state.mastered_count)
            
        st.progress((idx + 1) / len(cards))
        st.markdown("---")
        
        # HTML/CSS Card Display Container Box
        card_style = """
        <div style="background-color: {bg_color}; padding: 40px; border-radius: 12px; 
                    border: 2px solid #e0e0e0; min-height: 180px; display: flex; 
                    align-items: center; justify-content: center; text-align: center; 
                    box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px;">
            <h3 style="margin: auto; font-family: sans-serif; color: {text_color};">{content}</h3>
        </div>
        """
        
        if not st.session_state.is_flipped:
            st.markdown(card_style.format(bg_color="#ffffff", text_color="#1e293b", content=f"📝 {current_card['front']}"), unsafe_allow_html=True)
        else:
            st.markdown(card_style.format(bg_color="#f8fafc", text_color="#0f172a", content=f"💡 {current_card['back']}"), unsafe_allow_html=True)
            
        # Core Interaction Navigation Buttons
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 2, 1])
        with ctrl_col1:
            if st.button("⬅️ Previous", use_container_width=True):
                if st.session_state.card_index > 0:
                    st.session_state.card_index -= 1
                    st.session_state.is_flipped = False
                    st.rerun()
        with ctrl_col2:
            if st.button("🔄 Flip Card", use_container_width=True):
                st.session_state.is_flipped = not st.session_state.is_flipped
                st.rerun()
        with ctrl_col3:
            if st.button("Next ➡️", use_container_width=True):
                if st.session_state.card_index < len(cards) - 1:
                    st.session_state.card_index += 1
                    st.session_state.is_flipped = False
                    st.rerun()

        # Mastery Action Bar
        st.markdown(" ")
        if st.button("🎯 I Know This Concept!", use_container_width=True):
            st.session_state.mastered_count += 1
            if st.session_state.card_index < len(cards) - 1:
                st.session_state.card_index += 1
                st.session_state.is_flipped = False
            st.rerun()
    else:
        st.info("Your deck is currently empty. Chat with the AI to save custom cards, or auto-generate a deck above!")