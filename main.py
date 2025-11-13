import streamlit as st
import json
import os
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI
import openai
import dotenv
from notion_integration import NotionCategoryManager

dotenv.load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Notion manager
@st.cache_resource
def get_notion_manager():
    """Initialize and return Notion category manager"""
    try:
        return NotionCategoryManager()
    except Exception as e:
        st.error(f"Failed to initialize Notion: {e}")
        return None

# Load sentence transformer model for embeddings
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load categories from Notion
@st.cache_data(ttl=60)  # Cache for 60 seconds to reduce API calls
def load_categories(_notion_manager: Optional[NotionCategoryManager] = None):
    """Load categories from Notion or fallback to JSON"""
    if _notion_manager:
        try:
            return _notion_manager.load_categories()
        except Exception as e:
            st.warning(f"Could not load from Notion: {e}. Check your Notion configuration.")
            # Fallback to JSON if available
            try:
                with open('categories.json', 'r') as f:
                    return json.load(f)
            except FileNotFoundError:
                st.error("No categories found. Please configure Notion or create categories.json")
                return {"categories": {}}
    else:
        # Fallback to JSON
        try:
            with open('categories.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            st.error("Notion not configured and categories.json not found. Please set up Notion integration.")
            return {"categories": {}}

# Generate embeddings for all categories and subcategories
def get_category_embeddings(categories_data, _model):
    """Generate embeddings for all categories and subcategories"""
    embeddings = {}
    category_texts = {}
    
    for category, info in categories_data['categories'].items():
        # Create text representation for category
        category_text = f"{category}: {info['description']}"
        embeddings[category] = _model.encode(category_text)
        category_texts[category] = category_text
        
        # Create embeddings for subcategories
        for subcat in info['subcategories']:
            subcat_text = f"{category} > {subcat}: {info['description']}"
            key = f"{category} > {subcat}"
            embeddings[key] = _model.encode(subcat_text)
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

def expand_note_with_openai(text: str) -> str:
    """Use OpenAI to expand/improve the note for better context"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that improves notes by adding context and clarity. Expand the following note slightly to make it more meaningful and self-contained, while preserving the original meaning. Return only the improved note, nothing else."},
                {"role": "user", "content": text}
            ],
            max_tokens=500,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error expanding note: {e}")
        return text

def suggest_category_and_subcategory_from_text(text: str) -> Tuple[str, str]:
    """Use OpenAI to suggest both category and subcategory names based on the note content"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Based on the following note, suggest both a category name and subcategory name. Return in format: 'CategoryName > SubcategoryName'. Category should be 2-4 words, subcategory should be 1-3 words. Return ONLY the format 'Category > Subcategory', nothing else."},
                {"role": "user", "content": f"Suggest a category and subcategory for this note:\n\n{text}"}
            ],
            max_tokens=50,
            temperature=0.5
        )
        result = response.choices[0].message.content.strip()
        if " > " in result:
            parts = result.split(" > ", 1)
            return parts[0].strip(), parts[1].strip()
        return result, "General"
    except Exception as e:
        st.error(f"Error suggesting category name: {e}")
        return "New Category", "General"

def classify_note_multitopic(text: str, categories_data: Dict) -> List[Dict]:
    """Use OpenAI to classify note into multiple topics if needed"""
    try:
        # Create a list of available categories with subcategories
        category_list = []
        for category, info in categories_data['categories'].items():
            subcats = ', '.join(info['subcategories']) if info['subcategories'] else 'None'
            category_list.append(f"- {category} (subcategories: {subcats})")
        
        categories_str = '\n'.join(category_list)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"""You are a note classification assistant. Analyze the following note and determine if it contains multiple distinct topics. 

Available categories:
{categories_str}

IMPORTANT: Always return BOTH category and subcategory in the format "Category > Subcategory"
- If a subcategory exists, use it
- If no subcategory exists but category fits, suggest a new subcategory: "Category > NewSubcategoryName"
- If category doesn't fit, suggest: "NEW_CATEGORY: CategoryName > SubcategoryName"

If the note contains multiple distinct topics, split it into separate segments. For each segment:
1. Extract the relevant portion of text
2. Classify it into category and subcategory in format: "Category > Subcategory" or "NEW_CATEGORY: CategoryName > SubcategoryName"
3. Always include both category and subcategory separated by " > "

Return a JSON object with a "segments" array containing objects with this structure:
{{
  "segments": [
    {{
      "text_segment": "relevant portion of the note",
      "category": "Category > Subcategory" or "NEW_CATEGORY: CategoryName > SubcategoryName",
      "reasoning": "brief explanation"
    }}
  ]
}}

If the note is about a single topic, return one item in the segments array."""},
                {"role": "user", "content": f"Analyze and classify this note:\n\n{text}"}
            ],
            max_tokens=1000,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        # Handle JSON object response
        if isinstance(result, dict) and 'segments' in result:
            return result['segments']
        elif isinstance(result, list):
            return result
        elif isinstance(result, dict):
            # Single segment wrapped in object, convert to list
            return [result]
        else:
            return []
    except json.JSONDecodeError as e:
        # Fallback: try to parse as plain text
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"""You are a note classification assistant. Classify the following note into one of these categories:

{categories_str}

IMPORTANT: Always return BOTH category and subcategory in the format "Category > Subcategory"
- If a subcategory exists, use it
- If no subcategory exists but category fits, suggest a new subcategory: "Category > NewSubcategoryName"
- If category doesn't fit, suggest: "NEW_CATEGORY: CategoryName > SubcategoryName"

Return ONLY the category and subcategory in the format: "Category > Subcategory" or "NEW_CATEGORY: CategoryName > SubcategoryName"
Always include both category and subcategory separated by " > "."""},
                    {"role": "user", "content": f"Classify this note:\n\n{text}"}
                ],
                max_tokens=100,
                temperature=0.3
            )
            classification = response.choices[0].message.content.strip()
            return [{"text_segment": text, "category": classification, "reasoning": "Single topic note"}]
        except Exception as e2:
            st.error(f"Error classifying note: {e2}")
            return []
    except Exception as e:
        st.error(f"Error classifying note: {e}")
        return []

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

def suggest_categories_multitopic(note_text: str, categories_data: Dict, model, category_embeddings: Dict, expand_note: bool = False, similarity_threshold: float = 0.5) -> Dict:
    """Main function to suggest categories for a note, handling multiple topics"""
    results = {}
    
    # Optionally expand the note
    processed_text = note_text
    if expand_note:
        processed_text = expand_note_with_openai(note_text)
        results['expanded_note'] = processed_text
    
    # Multi-topic classification
    topic_segments = classify_note_multitopic(processed_text, categories_data)
    results['topic_segments'] = topic_segments
    
    # For each segment, also get embedding similarity
    for segment in topic_segments:
        segment_text = segment.get('text_segment', processed_text)
        segment_embedding = model.encode(segment_text)
        best_match, similarity_score = find_best_match_cosine_similarity(segment_embedding, category_embeddings)
        segment['embedding_match'] = best_match
        segment['similarity_score'] = float(similarity_score)
        
        # Check if similarity is too low and OpenAI didn't already suggest a new category
        category = segment.get('category', '')
        if similarity_score < similarity_threshold and not category.startswith("NEW_CATEGORY:"):
            # Suggest a new category based on low similarity
            suggested_category, suggested_subcategory = suggest_category_and_subcategory_from_text(segment_text)
            segment['category'] = f"NEW_CATEGORY: {suggested_category} > {suggested_subcategory}"
            segment['low_similarity_suggestion'] = True
            segment['similarity_reasoning'] = f"Similarity score ({similarity_score:.3f}) is below threshold ({similarity_threshold}). Best match was '{best_match}' but it's not a good fit."
    
    # Extract keywords from full note
    keywords = extract_keywords_with_openai(processed_text)
    results['keywords'] = keywords
    
    return results

def add_category_to_notion(category_name: str, description: str, subcategories: List[str], notion_manager: Optional[NotionCategoryManager], categories_data: Dict, update_state: bool = True):
    """Add a new category to Notion (or fallback to JSON)"""
    if notion_manager:
        try:
            success = notion_manager.add_category(category_name, description, subcategories)
            if success:
                # Update local state immediately
                categories_data['categories'][category_name] = {
                    "subcategories": subcategories,
                    "description": description
                }
                if update_state:
                    # Update session state immediately without reload
                    if 'categories_data' in st.session_state:
                        st.session_state.categories_data['categories'][category_name] = {
                            "subcategories": subcategories,
                            "description": description
                        }
                return categories_data
            else:
                st.error("Failed to add category to Notion")
                return categories_data
        except Exception as e:
            st.error(f"Error adding category to Notion: {e}")
            # Fallback to JSON
            return add_category_to_json_fallback(category_name, description, subcategories, categories_data)
    else:
        # Fallback to JSON
        return add_category_to_json_fallback(category_name, description, subcategories, categories_data)

def add_category_to_json_fallback(category_name: str, description: str, subcategories: List[str], categories_data: Dict):
    """Fallback: Add category to JSON file"""
    categories_data['categories'][category_name] = {
        "subcategories": subcategories,
        "description": description
    }
    try:
        with open('categories.json', 'w') as f:
            json.dump(categories_data, f, indent=2)
        load_categories.clear()
    except Exception as e:
        st.warning(f"Could not save to JSON: {e}")
    return categories_data

def add_subcategory_to_notion(category_name: str, subcategory_name: str, notion_manager: Optional[NotionCategoryManager], categories_data: Dict, update_state: bool = True):
    """Add a new subcategory to Notion (or fallback to JSON)"""
    if category_name not in categories_data['categories']:
        return False
    
    if subcategory_name in categories_data['categories'][category_name]['subcategories']:
        return True  # Already exists
    
    if notion_manager:
        try:
            success = notion_manager.add_subcategory(category_name, subcategory_name)
            if success:
                # Update local state immediately
                categories_data['categories'][category_name]['subcategories'].append(subcategory_name)
                if update_state:
                    # Update session state immediately without reload
                    if 'categories_data' in st.session_state:
                        if category_name in st.session_state.categories_data['categories']:
                            if subcategory_name not in st.session_state.categories_data['categories'][category_name]['subcategories']:
                                st.session_state.categories_data['categories'][category_name]['subcategories'].append(subcategory_name)
                return True
            else:
                st.error("Failed to add subcategory to Notion")
                return False
        except Exception as e:
            st.error(f"Error adding subcategory to Notion: {e}")
            # Fallback to JSON
            return add_subcategory_to_json_fallback(category_name, subcategory_name, categories_data)
    else:
        # Fallback to JSON
        return add_subcategory_to_json_fallback(category_name, subcategory_name, categories_data)

def add_subcategory_to_json_fallback(category_name: str, subcategory_name: str, categories_data: Dict):
    """Fallback: Add subcategory to JSON file"""
    if category_name in categories_data['categories']:
        if subcategory_name not in categories_data['categories'][category_name]['subcategories']:
            categories_data['categories'][category_name]['subcategories'].append(subcategory_name)
            try:
                with open('categories.json', 'w') as f:
                    json.dump(categories_data, f, indent=2)
                load_categories.clear()
                return True
            except Exception as e:
                st.warning(f"Could not save to JSON: {e}")
                return False
    return False

# Streamlit UI
def main():
    st.set_page_config(page_title="Note Categorizer", page_icon="📝", layout="wide")
    
    st.title("📝 Note Categorizer")
    st.markdown("Paste your note below and get AI-powered category suggestions!")
    
    # Initialize Notion manager
    notion_manager = get_notion_manager()
    
    # Check Notion configuration
    if notion_manager:
        # Test connection
        if 'notion_connected' not in st.session_state:
            st.session_state.notion_connected = notion_manager.test_connection()
        if not st.session_state.notion_connected:
            st.warning("⚠️ Notion connection test failed. Using fallback mode.")
    else:
        st.info("ℹ️ Notion not configured. Using local JSON fallback. Set NOTION_API_KEY and NOTION_PAGE_ID to use Notion.")
    
    # Load data
    try:
        model = load_embedding_model()
        categories_data = load_categories(notion_manager)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Initialize session state for results
    if 'categories_data' not in st.session_state:
        st.session_state.categories_data = categories_data
        st.session_state.notion_manager = notion_manager
    else:
        # Reload from Notion if needed (after category additions)
        categories_data = load_categories(notion_manager)
        st.session_state.categories_data = categories_data
        st.session_state.notion_manager = notion_manager
    
    # Generate embeddings with current categories (will be regenerated if categories change)
    # Use the latest categories_data from session state
    current_categories = st.session_state.categories_data if 'categories_data' in st.session_state else categories_data
    category_embeddings, category_texts = get_category_embeddings(current_categories, model)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ OPENAI_API_KEY not found in environment variables. Some features may not work.")
        st.info("Set it using: `export OPENAI_API_KEY='your-key-here'`")
    
    # Text input
    note_text = st.text_area(
        "Enter your note:",
        height=200,
        placeholder="Paste your note here...",
        help="The AI will analyze this text and suggest the best category. It can also split notes with multiple topics.",
        key="note_input"
    )
    auto_analyze = False
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        expand_note = st.checkbox("✨ Expand note", help="AI will add context to make the note more meaningful")
    with col2:
        save_to_notion = st.checkbox("💾 Save to Notion", help="Save note segments to corresponding Notion pages", disabled=notion_manager is None)
    with col3:
        analyze_button = st.button("🔍 Analyze & Classify", type="primary", use_container_width=True)
    
    # Initialize session state for results
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'similarity_threshold' not in st.session_state:
        st.session_state.similarity_threshold = 0.5
    
    # Check if we just added a category (skip analysis on rerun after adding)
    if 'category_just_added' in st.session_state:
        del st.session_state.category_just_added
        # Don't run analysis, just show previous results if any
        if 'analysis_results' in st.session_state and st.session_state.analysis_results:
            results = st.session_state.analysis_results
        else:
            results = None
    elif analyze_button and note_text.strip():
        with st.spinner("Analyzing your note..."):
            # Regenerate embeddings with latest categories
            current_categories = st.session_state.categories_data
            category_embeddings, _ = get_category_embeddings(current_categories, model)
            results = suggest_categories_multitopic(
                note_text, 
                current_categories, 
                model, 
                category_embeddings, 
                expand_note,
                st.session_state.similarity_threshold
            )
            st.session_state.analysis_results = results
    else:
        results = st.session_state.get('analysis_results', None)
    
    if results:
        
        # Display expanded note if applicable
        if 'expanded_note' in results and results['expanded_note'] != note_text:
            st.markdown("---")
            with st.expander("📝 Expanded Note (with added context)", expanded=True):
                st.text_area("", value=results['expanded_note'], height=150, disabled=True, key="expanded_display")
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Classification Results")
        
        if results and results.get('topic_segments'):
            num_topics = len(results['topic_segments'])
            if num_topics > 1:
                st.info(f"🔀 **Detected {num_topics} distinct topics in your note**")
            
            # Display each topic segment
            for idx, segment in enumerate(results['topic_segments'], 1):
                with st.container():
                    if num_topics > 1:
                        st.markdown(f"### Topic {idx}")
                    
                    category_str = segment.get('category', 'Unknown')
                    text_segment = segment.get('text_segment', note_text)
                    reasoning = segment.get('reasoning', '')
                    similarity_score = segment.get('similarity_score', 0)
                    is_low_similarity = segment.get('low_similarity_suggestion', False)
                    
                    # Parse category string to extract category and subcategory
                    is_new_category = category_str.startswith("NEW_CATEGORY:")
                    if is_new_category:
                        category_part = category_str.replace("NEW_CATEGORY:", "").strip()
                    else:
                        category_part = category_str
                    
                    # Split into category and subcategory
                    if " > " in category_part:
                        cat_name, subcat_name = category_part.split(" > ", 1)
                        cat_name = cat_name.strip()
                        subcat_name = subcat_name.strip()
                    else:
                        cat_name = category_part.strip()
                        subcat_name = None
                    
                    # Check if it's a new category or new subcategory
                    if is_new_category:
                        # Check if category exists but subcategory is new
                        category_exists = cat_name in st.session_state.categories_data.get('categories', {})
                        
                        if category_exists:
                            # Category exists, just need to add subcategory
                            st.warning(f"💡 **Suggested new subcategory: {cat_name} > {subcat_name}**")
                            st.caption(f"Category '{cat_name}' exists. Add subcategory '{subcat_name}'?")
                        else:
                            # New category and subcategory
                            st.warning(f"💡 **Suggested new category: {cat_name} > {subcat_name}**")
                        
                        # Show reasoning
                        if is_low_similarity and 'similarity_reasoning' in segment:
                            st.caption(f"⚠️ {segment['similarity_reasoning']}")
                        elif reasoning:
                            st.caption(f"Reasoning: {reasoning}")
                        
                        # Show similarity score if it triggered the suggestion
                        if is_low_similarity and 'similarity_score' in segment:
                            st.metric("Similarity Score", f"{similarity_score:.3f}", delta="Below threshold", delta_color="off")
                        
                        # Show text segment if different from full note
                        if text_segment != note_text:
                            with st.expander("📄 Relevant text segment"):
                                st.text(text_segment)
                        
                        # Confirmation to add
                        if category_exists:
                            # Just add subcategory
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.info(f"Add subcategory '{subcat_name}' to existing category '{cat_name}'?")
                            with col2:
                                if st.button("✅ Add Subcategory", key=f"add_subcat_{idx}", type="primary"):
                                    if add_subcategory_to_notion(cat_name, subcat_name, st.session_state.notion_manager, st.session_state.categories_data, update_state=True):
                                        st.success(f"✅ Added subcategory '{subcat_name}' to '{cat_name}'!")
                                        # Save note to Notion if enabled
                                        if save_to_notion and st.session_state.notion_manager:
                                            if st.session_state.notion_manager.save_note_to_page(cat_name, subcat_name, text_segment):
                                                st.info(f"💾 Note saved to {cat_name} > {subcat_name} in Notion")
                                        # Set flag to skip re-analysis on rerun
                                        st.session_state.category_just_added = True
                                        st.rerun()
                        else:
                            # Add new category with subcategory
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                new_cat_desc = st.text_input(
                                    f"Description for '{cat_name}':",
                                    placeholder="Enter a description for this category",
                                    key=f"new_cat_desc_{idx}"
                                )
                            with col2:
                                st.write("")  # Spacing
                                if st.button("✅ Add Category", key=f"add_cat_{idx}", type="primary"):
                                    # Add category with subcategory as first subpage
                                    updated_data = add_category_to_notion(
                                        cat_name,
                                        new_cat_desc or f"Notes about {cat_name}",
                                        [subcat_name] if subcat_name else [],
                                        st.session_state.notion_manager,
                                        st.session_state.categories_data,
                                        update_state=True
                                    )
                                    st.session_state.categories_data = updated_data
                                    st.success(f"✅ Added category '{cat_name}' with subcategory '{subcat_name}'!")
                                    # Save note to Notion if enabled
                                    if save_to_notion and st.session_state.notion_manager:
                                        if st.session_state.notion_manager.save_note_to_page(cat_name, subcat_name, text_segment):
                                            st.info(f"💾 Note saved to {cat_name} > {subcat_name} in Notion")
                                    # Set flag to skip re-analysis on rerun
                                    st.session_state.category_just_added = True
                                    st.rerun()
                    else:
                        # Existing category
                        display_str = f"{cat_name} > {subcat_name}" if subcat_name else cat_name
                        st.success(f"✅ **This segment fits under → {display_str}**")
                        if reasoning:
                            st.caption(f"Reasoning: {reasoning}")
                        
                        # Show text segment if different from full note
                        if text_segment != note_text:
                            with st.expander("📄 Relevant text segment"):
                                st.text(text_segment)
                        
                        # Save to Notion button if enabled
                        if save_to_notion and st.session_state.notion_manager:
                            if st.button("💾 Save to Notion", key=f"save_note_{idx}", help=f"Save this note segment to {display_str} in Notion"):
                                if st.session_state.notion_manager.save_note_to_page(cat_name, subcat_name, text_segment):
                                    st.success(f"✅ Note saved to {display_str} in Notion!")
                                else:
                                    st.error("Failed to save note to Notion")
                        
                        # Show similarity score
                        if 'similarity_score' in segment:
                            # Color code based on similarity
                            if similarity_score >= 0.7:
                                delta_color = "normal"
                            elif similarity_score >= 0.5:
                                delta_color = "off"
                            else:
                                delta_color = "inverse"
                            st.metric("Similarity Score", f"{similarity_score:.3f}", delta_color=delta_color)
        else:
            st.warning("Could not analyze the note. Please try again.")
        
        # Detailed results in expander
        with st.expander("🔬 Detailed Analysis", expanded=False):
            st.markdown("### Extracted Keywords")
            if results.get('keywords'):
                keywords_str = ", ".join(results['keywords'])
                st.info(keywords_str)
            else:
                st.warning("Keywords extraction unavailable")
            
            if results.get('topic_segments'):
                st.markdown("### Embedding Similarity Matches")
                for idx, segment in enumerate(results['topic_segments'], 1):
                    if 'embedding_match' in segment:
                        st.code(f"Topic {idx}: {segment['embedding_match']} (score: {segment.get('similarity_score', 0):.3f})")
    
    # Sidebar with category management
    with st.sidebar:
        st.header("⚙️ Settings")
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.similarity_threshold,
            step=0.05,
            help="If similarity score is below this threshold, a new category will be suggested. Lower = more new categories suggested."
        )
        st.session_state.similarity_threshold = similarity_threshold
        
        st.markdown("---")
        st.header("📁 Available Categories")
        for category, info in st.session_state.categories_data['categories'].items():
            with st.expander(category):
                st.caption(info['description'])
                st.markdown("**Subcategories:**")
                for subcat in info['subcategories']:
                    st.markdown(f"- {subcat}")
        
        st.markdown("---")
        st.header("➕ Add Category")
        
        with st.form("add_category_form"):
            new_category_name = st.text_input("Category Name", key="new_category_name")
            new_category_desc = st.text_area("Description", key="new_category_desc")
            new_subcategory_name = st.text_input(
                "Initial Subcategory (optional)",
                placeholder="Subcategory name",
                key="new_category_subcat"
            )
            submit_category = st.form_submit_button("Add Category", type="primary")
            
            if submit_category and new_category_name:
                subcats_list = [new_subcategory_name.strip()] if new_subcategory_name and new_subcategory_name.strip() else []
                updated_data = add_category_to_notion(
                    new_category_name,
                    new_category_desc or f"Notes about {new_category_name}",
                    subcats_list,
                    st.session_state.notion_manager,
                    st.session_state.categories_data
                )
                st.session_state.categories_data = updated_data
                # Reload to sync with Notion
                load_categories.clear()
                st.session_state.categories_data = load_categories(st.session_state.notion_manager)
                st.success(f"✅ Added category '{new_category_name}'!")
                st.rerun()
        
        st.markdown("---")
        st.header("➕ Add Subcategory")
        
        with st.form("add_subcategory_form"):
            parent_category = st.selectbox(
                "Parent Category",
                list(st.session_state.categories_data['categories'].keys()),
                key="parent_category_select"
            )
            new_subcategory_name = st.text_input("Subcategory Name", key="new_subcategory_name")
            submit_subcategory = st.form_submit_button("Add Subcategory", type="primary")
            
            if submit_subcategory and new_subcategory_name:
                if add_subcategory_to_notion(parent_category, new_subcategory_name, st.session_state.notion_manager, st.session_state.categories_data):
                    # Reload to sync with Notion
                    load_categories.clear()
                    st.session_state.categories_data = load_categories(st.session_state.notion_manager)
                    st.success(f"✅ Added '{new_subcategory_name}' to '{parent_category}'!")
                    st.rerun()
                else:
                    st.error(f"Subcategory already exists or category not found.")
        
        st.markdown("---")
        st.markdown("### How it works")
        st.markdown("""
        1. **Multi-topic Detection**: AI splits notes with multiple topics
        2. **Note Expansion**: Optionally adds context for better classification
        3. **Keyword Extraction**: Identifies key terms
        4. **Classification**: AI suggests categories/subcategories
        5. **Similarity Matching**: Compares with existing notes using embeddings
        """)

if __name__ == "__main__":
    main()
