import streamlit as st
import json
import os
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI
import openai

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load sentence transformer model for embeddings
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load categories from JSON
@st.cache_data
def load_categories():
    with open('categories.json', 'r') as f:
        return json.load(f)

# Generate embeddings for all categories and subcategories
@st.cache_data
def get_category_embeddings(categories_data, model):
    """Generate embeddings for all categories and subcategories"""
    embeddings = {}
    category_texts = {}
    
    for category, info in categories_data['categories'].items():
        # Create text representation for category
        category_text = f"{category}: {info['description']}"
        embeddings[category] = model.encode(category_text)
        category_texts[category] = category_text
        
        # Create embeddings for subcategories
        for subcat in info['subcategories']:
            subcat_text = f"{category} > {subcat}: {info['description']}"
            key = f"{category} > {subcat}"
            embeddings[key] = model.encode(subcat_text)
            category_texts[key] = subcat_text
    
    return embeddings, category_texts

def extract_keywords_with_openai(text: str) -> List[str]:
    """Use OpenAI to extract keywords from the text"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extract 5-10 key terms or topics from the following text. Return only a comma-separated list of keywords, nothing else."},
                {"role": "user", "content": text}
            ],
            max_tokens=100,
            temperature=0.3
        )
        keywords = response.choices[0].message.content.strip()
        return [k.strip() for k in keywords.split(',')]
    except Exception as e:
        st.error(f"Error extracting keywords: {e}")
        return []

def classify_note_with_openai(text: str, categories_data: Dict) -> Optional[str]:
    """Use OpenAI to classify the note into a category"""
    try:
        # Create a list of available categories
        category_list = []
        for category, info in categories_data['categories'].items():
            subcats = ', '.join(info['subcategories'])
            category_list.append(f"- {category} (subcategories: {subcats})")
        
        categories_str = '\n'.join(category_list)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"""You are a note classification assistant. Classify the following note into one of these categories and subcategories:

{categories_str}

Return ONLY the category and subcategory in the format: "Category > Subcategory"
If the note doesn't fit well into any existing category, return "NEW_CATEGORY" and suggest a new category name."""},
                {"role": "user", "content": f"Classify this note:\n\n{text}"}
            ],
            max_tokens=100,
            temperature=0.3
        )
        
        classification = response.choices[0].message.content.strip()
        return classification
    except Exception as e:
        st.error(f"Error classifying note: {e}")
        return None

def find_best_match_cosine_similarity(note_embedding: np.ndarray, category_embeddings: Dict) -> Tuple[str, float]:
    """Find the best matching category using cosine similarity"""
    best_match = None
    best_score = -1
    
    for category_key, cat_embedding in category_embeddings.items():
        # Calculate cosine similarity
        similarity = np.dot(note_embedding, cat_embedding) / (
            np.linalg.norm(note_embedding) * np.linalg.norm(cat_embedding)
        )
        
        if similarity > best_score:
            best_score = similarity
            best_match = category_key
    
    return best_match, best_score

def suggest_category(note_text: str, categories_data: Dict, model, category_embeddings: Dict) -> Dict:
    """Main function to suggest a category for a note"""
    results = {}
    
    # Method 1: OpenAI classification
    openai_classification = classify_note_with_openai(note_text, categories_data)
    results['openai_classification'] = openai_classification
    
    # Method 2: Cosine similarity with embeddings
    note_embedding = model.encode(note_text)
    best_match, similarity_score = find_best_match_cosine_similarity(note_embedding, category_embeddings)
    results['embedding_match'] = best_match
    results['similarity_score'] = float(similarity_score)
    
    # Method 3: Extract keywords
    keywords = extract_keywords_with_openai(note_text)
    results['keywords'] = keywords
    
    # Combine results - prefer OpenAI classification if it's not NEW_CATEGORY
    if openai_classification and openai_classification != "NEW_CATEGORY":
        final_suggestion = openai_classification
    else:
        final_suggestion = best_match
    
    results['final_suggestion'] = final_suggestion
    
    return results

# Streamlit UI
def main():
    st.set_page_config(page_title="Note Categorizer", page_icon="📝", layout="wide")
    
    st.title("📝 Note Categorizer")
    st.markdown("Paste your note below and get AI-powered category suggestions!")
    
    # Load data
    try:
        categories_data = load_categories()
        model = load_embedding_model()
        category_embeddings, category_texts = get_category_embeddings(categories_data, model)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ OPENAI_API_KEY not found in environment variables. Some features may not work.")
        st.info("Set it using: `export OPENAI_API_KEY='your-key-here'`")
    
    # Text input
    note_text = st.text_area(
        "Enter your note:",
        height=200,
        placeholder="Paste your note here...",
        help="The AI will analyze this text and suggest the best category"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button("🔍 Analyze & Classify", type="primary", use_container_width=True)
    
    if analyze_button and note_text.strip():
        with st.spinner("Analyzing your note..."):
            results = suggest_category(note_text, categories_data, model, category_embeddings)
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Classification Results")
        
        # Final suggestion (prominent)
        if results['final_suggestion']:
            if results['final_suggestion'] == "NEW_CATEGORY":
                st.success("✨ **This note doesn't fit existing categories well.**")
                st.info("💡 Consider creating a new category for this type of content.")
            else:
                st.success(f"✅ **This note fits under → {results['final_suggestion']}**")
        
        # Detailed results in expander
        with st.expander("🔬 Detailed Analysis", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### OpenAI Classification")
                if results['openai_classification']:
                    st.code(results['openai_classification'])
                else:
                    st.warning("OpenAI classification unavailable")
            
            with col2:
                st.markdown("### Embedding Similarity Match")
                if results['embedding_match']:
                    st.code(f"{results['embedding_match']}")
                    st.metric("Similarity Score", f"{results['similarity_score']:.3f}")
            
            st.markdown("### Extracted Keywords")
            if results['keywords']:
                keywords_str = ", ".join(results['keywords'])
                st.info(keywords_str)
            else:
                st.warning("Keywords extraction unavailable")
    
    # Sidebar with category information
    with st.sidebar:
        st.header("📁 Available Categories")
        for category, info in categories_data['categories'].items():
            with st.expander(category):
                st.caption(info['description'])
                st.markdown("**Subcategories:**")
                for subcat in info['subcategories']:
                    st.markdown(f"- {subcat}")
        
        st.markdown("---")
        st.markdown("### How it works")
        st.markdown("""
        1. **Keyword Extraction**: Uses OpenAI to identify key topics
        2. **Classification**: AI analyzes content and suggests category
        3. **Similarity Matching**: Compares with existing notes using embeddings
        4. **Final Suggestion**: Combines all methods for best result
        """)

if __name__ == "__main__":
    main()

