import os
# --- 1. CRITICAL MAC CRASH FIXES (Must be at the very top) ---
# This prevents the conflict between PyTorch and FAISS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import faiss
import pickle
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import CrossEncoder
from openai import OpenAI

# Force PyTorch to use 1 thread to avoid fighting with Streamlit
torch.set_num_threads(1)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & MODERN CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Smart Shopper",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Fancy" UI
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .product-card {
        background: rgba(40, 40, 45, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease-in-out;
        height: 380px; 
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .product-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
        border-color: #FF4B4B;
    }
    .img-container {
        width: 100%;
        height: 180px;
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 12px;
        background-color: #fff;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .img-container img {
        max-height: 100%;
        max-width: 100%;
        object-fit: contain;
    }
    .product-title {
        font-size: 15px;
        font-weight: 600;
        color: #f0f0f0;
        margin-bottom: 8px;
        line-height: 1.4;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .product-cat {
        font-size: 12px;
        color: #aaaaaa;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .buy-btn {
        display: block;
        width: 100%;
        text-align: center;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF9052 100%);
        color: white !important;
        padding: 10px;
        text-decoration: none;
        border-radius: 8px;
        font-weight: bold;
        font-size: 14px;
        transition: opacity 0.2s;
    }
    .buy-btn:hover {
        opacity: 0.9;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. CACHED RESOURCE LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_system_resources():
    """Loads Models, Indexes, and Dataframes only once."""
    resources = {}

    # A. Load Data
    filename = "Cleaned amazon dataset.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        if 'combined_text' not in df.columns:
            def create_rich_text(row):
                desc = str(row['full_description']) if pd.notna(row['full_description']) else ""
                return f"Product: {str(row['Product Name'])}\nCategory: {str(row['Category'])}\nDetails: {desc}"

            df['combined_text'] = df.apply(create_rich_text, axis=1)
        resources['df'] = df
    else:
        st.error(f"Error: File {filename} not found!")
        return None

    # B. Load AI Models
    device = "cpu"
    resources['device'] = device

    with st.spinner(f"Initializing AI Core (Device: {device})..."):
        model_name = "openai/clip-vit-base-patch32"
        resources['clip_model'] = CLIPModel.from_pretrained(model_name).to(device)
        resources['clip_processor'] = CLIPProcessor.from_pretrained(model_name)

        # Smart Reranker (BGE-Base)
        rerank_device = device
        resources['reranker'] = CrossEncoder('BAAI/bge-reranker-base', device=rerank_device)

    # C. Load FAISS Indexes
    if os.path.exists("text.index"):
        resources['text_index'] = faiss.read_index("text.index")
    else:
        resources['text_index'] = None

    if os.path.exists("image.index"):
        resources['image_index'] = faiss.read_index("image.index")
    else:
        resources['image_index'] = None

    if os.path.exists("valid_indices.pkl"):
        with open("valid_indices.pkl", "rb") as f:
            resources['valid_indices'] = pickle.load(f)
    else:
        resources['valid_indices'] = []

    return resources


sys_res = load_system_resources()


# -----------------------------------------------------------------------------
# 3. BACKEND LOGIC (UNIVERSAL HYBRID)
# -----------------------------------------------------------------------------

def retrieve_products(query, input_type="text", search_target="text_index", k=5):
    if sys_res['text_index'] is None: return pd.DataFrame()
    results = pd.DataFrame()
    device = sys_res['device']

    # 1. Generate Embedding
    if input_type == "text":
        inputs = sys_res['clip_processor'](text=[query], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            q_emb = sys_res['clip_model'].get_text_features(**inputs)
    elif input_type == "image":
        inputs = sys_res['clip_processor'](images=query, return_tensors="pt").to(device)
        with torch.no_grad():
            q_emb = sys_res['clip_model'].get_image_features(**inputs)

    q_emb = q_emb / q_emb.norm(p=2, dim=-1, keepdim=True)
    q_emb_np = q_emb.cpu().detach().numpy().astype('float32')

    # 2. Search Target Index
    if search_target == "text_index":
        D, I = sys_res['text_index'].search(q_emb_np, k)
        results = sys_res['df'].iloc[I[0]]

    elif search_target == "image_index":
        D, I = sys_res['image_index'].search(q_emb_np, k)
        if len(sys_res['valid_indices']) > 0:
            mapped_indices = [sys_res['valid_indices'][i] for i in I[0]]
            results = sys_res['df'].iloc[mapped_indices]

    return results


def generate_bot_response(user_query, image_input, api_key, chat_history):
    client = OpenAI(api_key=api_key)

    # --- 1. SMART HISTORY MANAGEMENT ---
    if image_input:
        history_str = "" # Hard Topic Switch on Image Upload
    else:
        history_str = ""
        for msg in chat_history[-4:]:
            history_str += f"{msg['role'].upper()}: {msg['content']}\n"

    def rewrite_query(user_input, history):
        system_prompt = (
            "You are a Search Query Optimizer. "
            "Your goal is to ensure the database finds relevant functional alternatives.\n"
            "### RULES:\n"
            "1. **Analyze Context:** \n"
            "   - If Follow-Up (e.g. 'cheaper'), merge with history.\n"
            "   - If New Topic, ignore history.\n"
            "2. **Remove Noise:** Delete polite phrases.\n"
            "3. **Output:** The final optimized search string ONLY (No tags)."
        )
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"CHAT HISTORY:\n{history}\n\nUSER INPUT: {user_input}"}
                ],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except:
            return user_input

    # --- 2. UNIVERSAL HYBRID RETRIEVAL (The Speed/Recall Fix) ---
    # Whether it's Text or Image input, we check BOTH indexes.
    # We cap 'k' at 12 per source to ensure total items to rerank is small (~24)
    # This keeps the BGE-Reranker speed acceptable (< 4s).
    
    K_PER_SOURCE = 12 
    
    if image_input:
        # A. Visual Search (Image Input)
        visual_matches = retrieve_products(image_input, input_type="image", search_target="image_index", k=K_PER_SOURCE)
        
        # B. Text Search (Text Input) - Finds "white", "cheap", etc.
        text_query = user_query if user_query.strip() else "product"
        text_matches = retrieve_products(text_query, input_type="text", search_target="text_index", k=K_PER_SOURCE)
        
        sys_prompt_type = "You are a visual AI assistant."
        context_intro = "I have analyzed the uploaded image and text."

    else:
        # A. Text Search (Standard)
        optimized_query = rewrite_query(user_query, history_str)
        text_matches = retrieve_products(optimized_query, input_type="text", search_target="text_index", k=K_PER_SOURCE)
        
        # B. Visual Search (via Text) - Catches "Skateboard" by shape if text fails
        visual_matches = retrieve_products(optimized_query, input_type="text", search_target="image_index", k=K_PER_SOURCE)
        
        sys_prompt_type = "You are a helpful sales assistant."
        context_intro = f"The user is searching for '{optimized_query}'."

    # --- 3. COMBINE & RERANK ---
    # Merge pools
    combined_candidates = pd.concat([visual_matches, text_matches])
    # Drop duplicates (don't rerank the same item twice)
    combined_candidates = combined_candidates.drop_duplicates(subset=['Product Name'])
    
    if not combined_candidates.empty:
        # Rerank using the Smart BGE Model
        # We use the text query (or optimized query) as the ground truth for relevance
        final_query = user_query if image_input else optimized_query
        
        product_texts = combined_candidates['combined_text'].tolist()
        pairs = [[final_query, text] for text in product_texts]
        
        scores = sys_res['reranker'].predict(pairs)
        combined_candidates['rerank_score'] = scores
        
        # Return Top 5
        retrieved_products = combined_candidates.sort_values('rerank_score', ascending=False).head(5)
    else:
        retrieved_products = pd.DataFrame()

    # --- 4. GENERATION (CONTEXT ISOLATION) ---
    sys_prompt = (
        f"{sys_prompt_type} Your goal is to recommend products based STRICTLY on the provided inventory.\n"
        "### CRITICAL RULES (Follow Strictly):\n"
        "1. **Current Inventory ONLY:** You must only recommend items listed in the '*** INVENTORY ***' section below. Do not use Chat History for items.\n"
        "2. **No Hallucinations:** If the inventory items do not match the user's request, ADMIT IT. Say 'I couldn't find exactly X, but here is Y'.\n"
        "3. **Strict Ordering:** Discuss the products in the exact order they appear in the Inventory (Item 1, then Item 2...).\n"
        "4. **Format:**\n"
        "   - **Item [X]:** [Product Name]\n"
        "   - **Why it fits:** [Brief explanation]\n"
        "   - **Image:** `![Product Name](Image_URL)`\n"
        "   - **Link:** `[View on Amazon](Product_URL)`"
    )

    # 5. Context Building
    if retrieved_products.empty:
        return "I couldn't find any relevant products in the database.", retrieved_products

    context_str = "*** INVENTORY (THE ONLY VALID ITEMS) ***\n"
    for i, row in retrieved_products.iterrows():
        raw_img_url = str(row['Image'])
        clean_img_url = raw_img_url.split("|")[0] if "|" in raw_img_url else raw_img_url

        context_str += f"Item {i + 1}:\n"
        context_str += f"- Name: {row['Product Name']}\n"
        context_str += f"- Link: {str(row['Product Url'])}\n"
        context_str += f"- Image_URL: {clean_img_url}\n"
        
        cat = str(row['Category']) if 'Category' in row else "General"
        desc = str(row['full_description']) if 'full_description' in row else str(row.get('About Product', ''))
        context_str += f"- Category: {cat}\n"
        context_str += f"- Description: {desc[:300]}...\n\n"

    # 6. Final Call
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"PREVIOUS CHAT HISTORY (Context Only, NO Products):\n{history_str}\n\nCURRENT INVENTORY (Select Products from HERE):\n{context_str}\n\n{context_intro}\nUser Question: {user_query}"}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content, retrieved_products
    except Exception as e:
        return f"Error communicating with OpenAI: {e}", retrieved_products


# -----------------------------------------------------------------------------
# 4. UI HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def display_product_grid(df):
    if df.empty:
        return

    st.markdown("### 🛍️ Recommended Products")

    cols = st.columns(3)

    for i, (idx, row) in enumerate(df.iterrows()):
        raw_img = str(row['Image'])
        img_url = raw_img.split("|")[0] if "|" in raw_img else raw_img

        category = str(row['Category']) if 'Category' in row else "Product"

        with cols[i % 3]:
            st.markdown(f"""
            <div class="product-card">
                <div class="img-container">
                    <img src="{img_url}" loading="lazy">
                </div>
                <div>
                    <div class="product-cat">{category}</div>
                    <div class="product-title">{row['Product Name']}</div>
                </div>
                <a href="{row['Product Url']}" target="_blank" class="buy-btn">View on Amazon</a>
            </div>
            """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. MAIN UI LAYOUT
# -----------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Control Panel")
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your SK- key here")

    col1, col2 = st.columns([1,2])
    if st.button("🗑️ Clear Chat", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.uploader_key += 1
        st.rerun()
        
    st.divider()

    st.markdown("### 🖼️ Visual Search")
    st.info("Upload an image here, then hit Enter in the chat box to search.")

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_file = st.file_uploader(
        "Upload Image...",
        type=["jpg", "png", "jpeg"],
        key=f"uploader_{st.session_state.uploader_key}"
    )

    current_image = None
    if uploaded_file:
        current_image = Image.open(uploaded_file).convert("RGB")
        st.success("Image Loaded! Press Enter in chat to send.")
        st.image(current_image, caption="Ready to Analyze", use_column_width=True)

st.title("🤖 AI Intelligent Shopper")
st.caption("Powered by CLIP + RAG + LLM (BGE-Reranker)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if "image_data" in message and message["image_data"]:
            st.image(message["image_data"], width=200)

        if "products" in message and not message["products"].empty:
            st.markdown("---")
            display_product_grid(message["products"])

prompt = st.chat_input("Ask about a product or describe what you need...")

if prompt:
    if not api_key:
        st.warning("Please enter your OpenAI API Key in the sidebar first.")
        st.stop()

    user_text = prompt
    user_image = current_image if current_image else None

    with st.chat_message("user"):
        if user_image:
            st.image(user_image, width=200)
        st.markdown(user_text)

    st.session_state.messages.append({
        "role": "user",
        "content": user_text,
        "image_data": user_image
    })

    with st.chat_message("assistant"):
        with st.spinner("Analyzing intent and searching inventory..."):

            response_text, products_df = generate_bot_response(
                user_text,
                user_image,
                api_key,
                st.session_state.messages
            )

            st.markdown(response_text)

            if not products_df.empty:
                st.markdown("---")
                display_product_grid(products_df)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "products": products_df
    })

    if user_image is not None:
        st.session_state.uploader_key += 1
        st.rerun()
