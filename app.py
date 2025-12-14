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

        rerank_device = device 
        resources['reranker'] = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=rerank_device)

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
# 3. BACKEND LOGIC
# -----------------------------------------------------------------------------

# --- UPDATED: SEQUENTIAL RERANKING LOGIC ---
def retrieve_products(query, input_type="text", search_target="text_index", k=5, text_modifier=None):
    if sys_res['text_index'] is None: return pd.DataFrame()

    results = pd.DataFrame()
    device = sys_res['device']

    # --- SCENARIO 1: TEXT ONLY (Standard Search) ---
    if input_type == "text":
        inputs = sys_res['clip_processor'](text=[query], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            q_emb = sys_res['clip_model'].get_text_features(**inputs)
        
        q_emb = q_emb / q_emb.norm(p=2, dim=-1, keepdim=True)
        q_emb_np = q_emb.cpu().detach().numpy().astype('float32')
        
        if search_target == "text_index":
            D, I = sys_res['text_index'].search(q_emb_np, k)
            results = sys_res['df'].iloc[I[0]]
        elif search_target == "image_index":
            D, I = sys_res['image_index'].search(q_emb_np, k)
            if len(sys_res['valid_indices']) > 0:
                mapped_indices = [sys_res['valid_indices'][i] for i in I[0]]
                results = sys_res['df'].iloc[mapped_indices]

    # --- SCENARIO 2: IMAGE INPUT (With optional Text Modifier) ---
    elif input_type == "image":
        # Step 1: Get Image Embedding
        inputs = sys_res['clip_processor'](images=query, return_tensors="pt").to(device)
        with torch.no_grad():
            img_emb = sys_res['clip_model'].get_image_features(**inputs)
            img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
        
        q_emb_np = img_emb.cpu().detach().numpy().astype('float32')

        # Step 2: Initial Visual Search (Get Top 50 candidates)
        # We fetch a larger pool to increase the odds of finding the specific color/variant
        k_candidates = 50 
        D, I = sys_res['image_index'].search(q_emb_np, k_candidates)

        if len(sys_res['valid_indices']) > 0:
            mapped_indices = [sys_res['valid_indices'][i] for i in I[0]]
            candidate_df = sys_res['df'].iloc[mapped_indices].copy()
        else:
            return pd.DataFrame()

        # Step 3: APPLY TEXT MODIFIER (Sequential Reranking)
        # If user provided text (e.g., "white"), we resort the 50 candidates by that text
        if text_modifier and len(text_modifier.strip()) > 0:
            
            # We use the CrossEncoder (Reranker) to score how well the 
            # text descriptions of the 50 visual matches align with the user's text constraint.
            
            product_texts = candidate_df['combined_text'].tolist()
            # Create pairs for the CrossEncoder: [("white", "Bear Description 1"), ("white", "Bear Description 2")...]
            pairs = [[text_modifier, text] for text in product_texts]
            
            # Predict similarity scores
            scores = sys_res['reranker'].predict(pairs)
            candidate_df['rerank_score'] = scores
            
            # Sort by the new score (Text Match) and take top k
            results = candidate_df.sort_values('rerank_score', ascending=False).head(k)
            
        else:
            # No text provided, just return top k visual matches
            results = candidate_df.head(k)

    return results


def retrieve_and_rerank(query, mode="functional", k_final=5, k_initial=50):
    if mode == "visual":
        return retrieve_products(query, input_type="text", search_target="image_index", k=k_final)
    else:
        initial_results = retrieve_products(query, input_type="text", search_target="text_index", k=k_initial)
        
    if initial_results.empty: return initial_results

    product_texts = initial_results['combined_text'].tolist()
    pairs = [[query, text] for text in product_texts]

    scores = sys_res['reranker'].predict(pairs)
    ranked_results = initial_results.copy()
    ranked_results['rerank_score'] = scores

    return ranked_results.sort_values('rerank_score', ascending=False).head(k_final)


def generate_bot_response(user_query, image_input, api_key, chat_history):
    client = OpenAI(api_key=api_key)

    history_str = ""
    for msg in chat_history[-4:]:
        history_str += f"{msg['role'].upper()}: {msg['content']}\n"

    def rewrite_query(user_input, history):
        system_prompt = (
            "You are a Search Query Optimizer. "
            "Your goal is to ensure the database finds relevant functional alternatives based on User Input and History.\n"
            "### INSTRUCTIONS:\n"
            "1. **Check for Topic Switch (CRITICAL):** Look at the 'Chat History'. If the user's new input implies a new topic (e.g., switched from 'skateboards' to 'rabbits'), IGNORE the old topic. Recency determines the topic.\n"
            "2. **Remove Noise:** Delete polite phrases (e.g., 'show me', 'can I get').\n"
            "3. **Expand Brands to Categories:**\n"
            "   - Input: 'AirPods' -> Output: 'AirPods Earbuds Headphones In-Ear'\n"
            "   - Input: 'iPhone' -> Output: 'iPhone Smartphone Mobile'\n"
            "   - Input: 'GoPro' -> Output: 'GoPro Action Camera Video'\n"
            "4. **Expand Categories:**\n"
            "   - Input: 'Skateboard' -> Output: 'Skateboard Longboard Cruiser Deck'\n"
            "5. **(NEW) DETECT VISUAL INTENT:** If the user describes appearance (e.g., 'looks like', 'red', 'modern style', 'shape'), prepend `[VISUAL]` to your output. Otherwise, prepend `[FUNCTIONAL]`.\n"
            "6. Output ONLY the final optimized search string."
        )

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"CHAT HISTORY (Oldest to Newest):\n{history}\n\nUSER INPUT: {user_input}"}
                ],
                temperature=0
            )
            cleaned_query = response.choices[0].message.content.strip()
            print(f"DEBUG: Expanded Query: '{user_input}' -> '{cleaned_query}'")
            return cleaned_query
        except:
            return f"[FUNCTIONAL] {user_input}"

    # 1. Retrieval
    if image_input:
        # Use the Rewriter to get the "clean" constraint text (e.g. "white")
        # This helps the Reranker work better by removing "I want a..."
        clean_text_modifier = rewrite_query(user_query, history_str).replace("[VISUAL]", "").replace("[FUNCTIONAL]", "").strip()

        # Call the updated sequential retrieval
        retrieved_products = retrieve_products(
            image_input, 
            input_type="image", 
            search_target="image_index", 
            k=5, 
            text_modifier=clean_text_modifier
        )
        
        sys_prompt = (
            "You are a visual AI assistant. The user has uploaded an EXTERNAL reference image "
            "and may have provided text constraints (e.g., 'white'). "
            "The products listed below are VISUALLY SIMILAR recommendations found in the database. "
            "If the user provided a text constraint (like color or material), the system has attempted to filter for that. "
            "Your job is to explain why these items are good matches based on the visual and text inputs."
        )
        context_intro = "I analyzed the uploaded image and text constraints. Here are the best matches from our inventory:"
    else:
        optimized_query_raw = rewrite_query(user_query, history_str)
        
        if "[VISUAL]" in optimized_query_raw:
            search_mode = "visual"
            optimized_query = optimized_query_raw.replace("[VISUAL]", "").strip()
        else:
            search_mode = "functional"
            optimized_query = optimized_query_raw.replace("[FUNCTIONAL]", "").strip()

        retrieved_products = retrieve_and_rerank(optimized_query, mode=search_mode, k_final=5, k_initial=50)

        sys_prompt = (
            "You are a helpful sales assistant. Your goal is to recommend the best products based on function.\n"
            "### RULES:\n"
            "1. **Determine Intent:**\n"
            "   - **Broad:** (e.g., 'party'). Recommend **3 distinct options**.\n"
            "   - **Specific:** (e.g., 'AirPods'). Find that item or the closest **functional** alternative.\n"
            "2. **Relevance Hierarchy (Follow Strictly):**\n"
            "   - **Tier 1 (Direct Match):** 'AirPods', 'Headphones', 'Earbuds'. -> SHOW THESE FIRST.\n"
            "   - **Tier 2 (Functional Fallback):** If NO Headphones exist, look for **Audio Gear** (e.g., 'Speakers', 'Soundbars'). -> SHOW THESE with a note ('I don't have headphones, but here are some speakers...').\n"
            "   - **Tier 3 (Irrelevant):** 'Drone', 'Air Filter', 'Airplane'. -> **NEVER SHOW THESE** for headphone requests.\n"
            "3. **STRICT FORMATTING:**\n"
            "   - **Name:** [Exact Product Name]\n"
            "   - **Description:** [Description]\n"
            "   - **Image:** `![Product Name](Image_URL)`\n" 
            "   - **Link:** `[View on Amazon](Product_URL)`"
        )
        context_intro = (
            f"The user is searching for '{optimized_query}' (Original input: {user_query}). "
            "Scan the inventory below. "
            "Prioritize FUNCTION over text matches (e.g., reject 'Air Drones' if user wants 'AirPods')."
        )

    # 2. Context
    if retrieved_products.empty:
        return "I couldn't find any relevant products in the database.", retrieved_products

    context_str = "*** INVENTORY ***\n"
    for i, row in retrieved_products.iterrows():
        context_str += f"Item {i + 1}:\n"
        context_str += f"- Name: {row['Product Name']}\n"
        context_str += f"- Link: {str(row['Product Url'])}\n"

        raw_img_url = str(row['Image'])
        if "|" in raw_img_url:
            clean_img_url = raw_img_url.split("|")[0]
        else:
            clean_img_url = raw_img_url
        context_str += f"- Image_URL: {clean_img_url}\n"

        cat = str(row['Category']) if 'Category' in row else "General"
        desc = str(row['full_description']) if 'full_description' in row else str(row.get('About Product', ''))
        context_str += f"- Category: {cat}\n"
        context_str += f"- Description: {desc[:300]}...\n\n"

    # 3. Generation
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Previous Chat History:\n{history_str}\n\nContext:\n{context_str}\n\n{context_intro}\nUser Question: {user_query}"}
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
st.caption("Powered by CLIP + RAG + LLM")

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
