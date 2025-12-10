import os
# --- 1. CONFIG & SETUP ---
# Fixes for local Mac/Windows setups (harmless on Cloud)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import faiss
import pickle
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import CrossEncoder
from openai import OpenAI

# -----------------------------------------------------------------------------
# 2. PAGE CONFIG & MODERN CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Smart Shopper",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Fancy" UI (Glassmorphism + Hover Effects)
st.markdown("""
<style>
    /* Global Background tweaks (optional, keeps Streamlit dark mode) */

    /* Chat Message Bubble Styling */
    .stChatMessage {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* THE PRODUCT CARD - The core of the fancy UI */
    .product-card {
        background: rgba(40, 40, 45, 0.6); /* Semi-transparent dark */
        backdrop-filter: blur(10px);        /* Glass blur effect */
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease-in-out;
        height: 380px; /* Fixed height for alignment */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    /* Hover Effect: Lift up and Glow */
    .product-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
        border-color: #FF4B4B; /* Streamlit Red Accent */
    }

    /* Image Container to keep images proportional */
    .img-container {
        width: 100%;
        height: 180px;
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 12px;
        background-color: #fff; /* White bg for product images */
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .img-container img {
        max-height: 100%;
        max-width: 100%;
        object-fit: contain;
    }

    /* Typography */
    .product-title {
        font-size: 15px;
        font-weight: 600;
        color: #f0f0f0;
        margin-bottom: 8px;
        line-height: 1.4;
        /* Truncate text after 2 lines */
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

    /* Buy Button */
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
# 3. CACHED RESOURCE LOADING
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resources['device'] = device

    with st.spinner(f"Initializing AI Core (Device: {device})..."):
        model_name = "openai/clip-vit-base-patch32"
        resources['clip_model'] = CLIPModel.from_pretrained(model_name).to(device)
        resources['clip_processor'] = CLIPProcessor.from_pretrained(model_name)

        rerank_device = device if device != "mps" else "cpu"
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
# 4. BACKEND LOGIC
# -----------------------------------------------------------------------------

def retrieve_products(query, modality="text", k=5):
    if sys_res['text_index'] is None: return pd.DataFrame()

    results = pd.DataFrame()
    device = sys_res['device']

    if modality == "text":
        inputs = sys_res['clip_processor'](text=[query], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            q_emb = sys_res['clip_model'].get_text_features(**inputs)
        q_emb = q_emb / q_emb.norm(p=2, dim=-1, keepdim=True)
        q_emb = q_emb.cpu().numpy()

        D, I = sys_res['text_index'].search(q_emb, k)
        results = sys_res['df'].iloc[I[0]]

    elif modality == "image":
        inputs = sys_res['clip_processor'](images=query, return_tensors="pt").to(device)
        with torch.no_grad():
            q_emb = sys_res['clip_model'].get_image_features(**inputs)
        q_emb = q_emb / q_emb.norm(p=2, dim=-1, keepdim=True)
        q_emb = q_emb.cpu().numpy()

        D, I = sys_res['image_index'].search(q_emb, k)

        if len(sys_res['valid_indices']) > 0:
            mapped_indices = [sys_res['valid_indices'][i] for i in I[0]]
            results = sys_res['df'].iloc[mapped_indices]

    return results


def retrieve_and_rerank(query, k_final=5, k_initial=50):
    initial_results = retrieve_products(query, modality="text", k=k_initial)
    if initial_results.empty: return initial_results

    product_texts = initial_results['combined_text'].tolist()
    pairs = [[query, text] for text in product_texts]

    scores = sys_res['reranker'].predict(pairs)
    ranked_results = initial_results.copy()
    ranked_results['rerank_score'] = scores

    return ranked_results.sort_values('rerank_score', ascending=False).head(k_final)


def generate_bot_response(user_query, image_input, api_key, chat_history): # Added chat_history
    client = OpenAI(api_key=api_key)

    def rewrite_query(user_input):
        """
        Standardizes and EXPANDS the user query to ensure the database finds relevant items.
        """
        system_prompt = (
            "You are a Search Query Optimizer. "
            "Your goal is to ensure the database finds relevant functional alternatives.\n"
            "### INSTRUCTIONS:\n"
            "1. **Remove Noise:** Delete polite phrases (e.g., 'show me', 'can I get').\n"
            "2. **Expand Brands to Categories:**\n"
            "   - Input: 'AirPods' -> Output: 'AirPods Earbuds Headphones In-Ear'\n"
            "   - Input: 'iPhone' -> Output: 'iPhone Smartphone Mobile'\n"
            "   - Input: 'GoPro' -> Output: 'GoPro Action Camera Video'\n"
            "3. **Expand Categories:**\n"
            "   - Input: 'Skateboard' -> Output: 'Skateboard Longboard Cruiser Deck'\n"
            "4. Output ONLY the final optimized search string."
        )

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0
            )
            cleaned_query = response.choices[0].message.content.strip()
            print(f"DEBUG: Expanded Query: '{user_input}' -> '{cleaned_query}'")
            return cleaned_query
        except:
            return user_input

    # 1. Retrieval
    if image_input:
        retrieved_products = retrieve_products(image_input, modality="image", k=5)
        sys_prompt = "You are a helpful visual AI assistant. The user has uploaded an image and asked a question.\n### INSTRUCTIONS:\n1. **Answer First:** Identify the object and answer the user's specific question.\n2. **Recommend Second:** List matching products from the context.\n3. **STRICT FORMATTING:**\n   - **Name:** [Exact Product Name]\n   - **Image:** `![Product Name](Image_URL)`\n   - **Link:** `[Click to Buy](Product_URL)`\n"
        context_intro = "I analyzed the image you uploaded. Here are the most visually similar products:"
    else:
        # retrieved_products = retrieve_and_rerank(user_query, k_final=5)
        optimized_query = rewrite_query(user_query)
        retrieved_products = retrieve_and_rerank(optimized_query, k_final=5, k_initial=50)
        # sys_prompt = "You are a helpful sales assistant. Recommend best products based on function and relevance."
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
            "   - **Image:** `![Product Name](Image_URL)`\n"
            "   - **Link:** `[Click to Buy](Product_URL)`"
        )
        context_intro = (
            f"The user is searching for '{user_query}'. "
            "Scan the inventory below. "
            "Prioritize FUNCTION over text matches (e.g., reject 'Air Drones' if user wants 'AirPods')."
        )

    # 2. Context
    if retrieved_products.empty:
        return "I couldn't find any relevant products in the database.", retrieved_products

    context_str = "*** INVENTORY CANDIDATES ***\n"
    for i, row in retrieved_products.iterrows():
        context_str += f"Item {i+1}:\n"
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

    # 3. Generation (Updated with Memory)
    
    # Construct message list with history
    messages_payload = [{"role": "system", "content": sys_prompt}]
    
    for msg in chat_history:
        if msg["role"] in ["user", "assistant"]:
            messages_payload.append({"role": msg["role"], "content": msg["content"]})
            
    # Add current query with context
    messages_payload.append({"role": "user", "content": f"Context:\n{context_str}\n\n{context_intro}\nUser Input: {user_query}"})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages_payload,
            temperature=0.3
        )
        return response.choices[0].message.content, retrieved_products
    except Exception as e:
        return f"Error communicating with OpenAI: {e}", retrieved_products


# -----------------------------------------------------------------------------
# 5. UI HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def display_product_grid(df):
    """
    Renders the products in a perfect 3-column grid.
    Fixes the vertical stacking issue.
    """
    if df.empty:
        return

    st.markdown("### 🛍️ Recommended Products")

    # Create the columns OUTSIDE the loop
    cols = st.columns(3)

    for i, (idx, row) in enumerate(df.iterrows()):
        # Handle Amazon multi-images (split by |)
        raw_img = str(row['Image'])
        img_url = raw_img.split("|")[0] if "|" in raw_img else raw_img

        category = str(row['Category']) if 'Category' in row else "Product"

        # Use HTML/CSS for the Fancy Card
        # The key is using cols[i % 3] to cycle through columns 0, 1, 2
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
# 6. MAIN UI LAYOUT
# -----------------------------------------------------------------------------

# --- Sidebar: Settings & Upload ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your SK- key here")

    st.divider()
    
    # --- UPDATE: Initialize Uploader Key ---
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    st.markdown("### 🖼️ Visual Search")
    st.info("Upload an image here, then hit Enter in the chat box to search.")

    # --- UPDATE: Use Dynamic Key ---
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

# --- Main Chat Area ---
st.title("🤖 AI Intelligent Shopper")
st.caption("Powered by CLIP + RAG + LLM")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if "image_data" in message and message["image_data"]:
            st.image(message["image_data"], width=200)

        if "products" in message and not message["products"].empty:
            st.markdown("---")
            # CALL THE GRID FUNCTION TO ENSURE HORIZONTAL LAYOUT
            display_product_grid(message["products"])

# --- Fixed Input at Bottom ---
prompt = st.chat_input("Ask about a product or describe what you need...")

if prompt:
    if not api_key:
        st.warning("Please enter your OpenAI API Key in the sidebar first.")
        st.stop()

    user_text = prompt
    user_image = current_image if current_image else None

    # Show User Message
    with st.chat_message("user"):
        if user_image:
            st.image(user_image, width=200)
        st.markdown(user_text)

    # Save to History
    st.session_state.messages.append({
        "role": "user",
        "content": user_text,
        "image_data": user_image
    })

    # AI Generation
    with st.chat_message("assistant"):
        with st.spinner("Analyzing intent and searching inventory..."):
            # --- UPDATE: Pass Chat History ---
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

    # Save Assistant Response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "products": products_df
    })
    
    # --- UPDATE: Reset Image and Rerun ---
    st.session_state.uploader_key += 1
    st.rerun()

