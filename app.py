import os
import streamlit as st
from dotenv import load_dotenv
import numpy as np

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from huggingface_hub import InferenceClient

# Load environment variables if .env file exists
load_dotenv()

st.set_page_config(page_title="RAG Chatbot", layout="wide")


class HuggingFaceAPIEmbeddings(Embeddings):
    """Custom embeddings class using HuggingFace Hub InferenceClient."""
    
    def __init__(self, api_key: str, model_name: str):
        self.client = InferenceClient(token=api_key)
        self.model_name = model_name
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        embeddings = []
        for text in texts:
            try:
                # Use feature_extraction which returns embeddings
                result = self.client.feature_extraction(text, model=self.model_name)
                
                # Convert to list if it's a numpy array
                if isinstance(result, np.ndarray):
                    embeddings.append(result.tolist())
                else:
                    embeddings.append(result)
                    
            except Exception as e:
                st.error(f"Embedding error for text: {text[:50]}... | Error: {e}")
                raise
        
        return embeddings
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self.embed_documents([text])[0]


st.title("🤖 RAG Chatbot")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    hf_token = os.getenv("HF_TOKEN", "")
    # Model selection
    embedding_model = st.selectbox(
        "Embedding Model",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-small-en-v1.5",
            "sentence-transformers/all-mpnet-base-v2"
        ],
        help="Lightweight models that run on HuggingFace's servers"
    )
    
    llm_model = st.selectbox(
        "LLM Model",
        [
            "meta-llama/Llama-3.2-3B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "HuggingFaceH4/zephyr-7b-beta",
            "microsoft/Phi-3-mini-4k-instruct",
            "google/gemma-2-2b-it"
        ],
        help="Language model for generating answers (chat-optimized models)"
    )
    
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
    num_results = st.slider("Number of Retrieved Documents", 1, 5, 3)
    
    st.markdown("### Knowledge Base")
    st.info("Ensure your documents are in the `knowledge_base` folder.")
    
    if st.button("🔄 Reload Knowledge Base"):
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📋 Setup Instructions")
    st.markdown(
        "1. Go to [HuggingFace](https://huggingface.co/settings/tokens)\n"
        "2. Create **Fine-grained** token\n"
        "3. ✅ Enable **'Make calls to Inference Providers'**\n"
        "4. Copy and paste token above"
    )

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Function to load and process knowledge base
@st.cache_resource(show_spinner="Loading Knowledge Base...")
def load_and_process_data(_hf_token, _embedding_model, _chunk_size):
    """Load documents and create vector store using API-based embeddings."""
    
    if not os.path.exists("knowledge_base"):
        os.makedirs("knowledge_base")
        st.error("Created 'knowledge_base' folder. Please add some .txt files and refresh.")
        st.stop()

    # Load documents
    try:
        loader = DirectoryLoader(
            "knowledge_base", 
            glob="**/*.txt", 
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True}
        )
        documents = loader.load()
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        st.stop()

    if not documents:
        st.error("No documents found in 'knowledge_base'. Please add .txt files.")
        st.stop()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_chunk_size, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # Create embeddings using custom class
    embeddings = HuggingFaceAPIEmbeddings(
        api_key=_hf_token,
        model_name=_embedding_model
    )

    # Test the embeddings first
    try:
        st.info("Testing embedding API connection...")
        test_embedding = embeddings.embed_query("test")
        st.success(f"✅ Embedding API working! Vector size: {len(test_embedding)}")
    except Exception as e:
        st.error(f"❌ Embedding API test failed: {e}")
        st.error(
            "**Please check:**\n"
            "1. Your token has 'Make calls to Inference Providers' enabled\n"
            "2. You're using a 'Fine-grained' or 'Write' token type\n"
            "3. The token is correctly copied (no extra spaces)\n"
            "4. The model is available on HuggingFace"
        )
        st.stop()

    # Create vector store
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore, len(documents), len(chunks)


def generate_answer(query: str, context: str, token: str, model: str) -> str:
    """Use HuggingFace Inference API to generate an answer."""
    
    client = InferenceClient(token=token)

    # Build system message and user message for chat completion
    system_message = "You are a helpful AI assistant. Answer questions based ONLY on the provided context. If the answer is not in the context, say 'I cannot find this information in the provided documents'."
    
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    try:
        # Use chat_completion which works with most modern models
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        response = client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=512,
            temperature=0.2,
            top_p=0.9,
        )
        
        # Extract the response text
        if hasattr(response, 'choices') and len(response.choices) > 0:
            answer = response.choices[0].message.content.strip()
            return answer if answer else "⚠️ Model returned empty response"
        else:
            return "⚠️ Unexpected response format"
        
    except Exception as e:
        error_msg = str(e).lower()
        
        if "503" in error_msg or "loading" in error_msg:
            return "⚠️ Model is currently loading. Please wait 20-30 seconds and try again."
        elif "401" in error_msg or "unauthorized" in error_msg:
            return "⚠️ Authentication failed. Please check your HuggingFace token."
        elif "403" in error_msg or "forbidden" in error_msg:
            return "⚠️ Access forbidden. Make sure 'Make calls to Inference Providers' is enabled."
        elif "timeout" in error_msg:
            return "⚠️ Request timed out. Please try again."
        elif "not supported" in error_msg:
            return f"⚠️ This model doesn't support chat completion. Try selecting a different model from the sidebar."
        else:
            return f"⚠️ Error: {str(e)}"


# Main Application Logic
if not hf_token:
    st.warning("⚠️ Please enter your HuggingFace token in the sidebar.")
    st.info(
        "### 🔑 How to Get Your Token:\n\n"
        "1. Visit [HuggingFace Settings](https://huggingface.co/settings/tokens)\n"
        "2. Click **'Create new token'**\n"
        "3. Select **'Fine-grained'** token type\n"
        "4. ✅ Check **'Make calls to Inference Providers'**\n"
        "5. Create and copy your token\n"
        "6. Paste it in the sidebar ⬅️"
    )
    st.stop()

try:
    # Load knowledge base
    vector_store, num_docs, num_chunks = load_and_process_data(
        hf_token, 
        embedding_model, 
        chunk_size
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": num_results})
    
    # Show knowledge base stats
    st.success(f"✅ Knowledge base loaded: {num_docs} documents, {num_chunks} chunks")

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if user_input := st.chat_input("Ask something about your knowledge base..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                relevant_docs = retriever.invoke(user_input)

            if relevant_docs:
                context = "\n\n".join(
                    [f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(relevant_docs)]
                )

                with st.spinner("Generating answer..."):
                    response = generate_answer(user_input, context, hf_token, llm_model)
                
                st.markdown(response)
                
                with st.expander("📄 View Source Documents"):
                    for i, doc in enumerate(relevant_docs):
                        source_file = doc.metadata.get('source', 'Unknown')
                        st.markdown(f"**Document {i+1}** (from `{os.path.basename(source_file)}`):")
                        st.text(doc.page_content)
                        st.markdown("---")
            else:
                response = "❌ No relevant documents found."
                st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

except Exception as e:
    st.error(f"❌ Error: {e}")
    
    error_str = str(e).lower()
    
    if "403" in error_str or "forbidden" in error_str:
        st.error(
            "### 🔑 Token Permission Issue\n\n"
            "This error usually means your token doesn't have the right permissions.\n\n"
            "**Fix:**\n"
            "1. Go to https://huggingface.co/settings/tokens\n"
            "2. **Delete** your old token\n"
            "3. Create a **NEW** token:\n"
            "   - Type: **Fine-grained**\n"
            "   - ✅ Check **'Make calls to Inference Providers'**\n"
            "4. Copy the NEW token\n"
            "5. Paste it in the sidebar and refresh"
        )
    elif "410" in error_str or "gone" in error_str:
        st.error(
            "### ⚠️ API Endpoint Issue\n\n"
            "The API endpoint has changed or the model is no longer available.\n\n"
            "**Try:**\n"
            "1. Select a different embedding model from the sidebar\n"
            "2. Make sure you have the latest version: `pip install --upgrade huggingface_hub`\n"
            "3. Check if the model exists on HuggingFace"
        )
    
    with st.expander("🐛 Full Error Details"):
        st.exception(e)

# Footer
st.markdown("---")
st.caption("💡 All processing via HuggingFace API - no local model downloads!")