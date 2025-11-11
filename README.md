# MindEnhance - Note Categorizer

An AI-powered note categorization tool that analyzes your notes and automatically suggests the best folder/category (like "Python > Decorators", "Career > Interview Prep", or "Machine Learning > Neural Networks") based on content.

## Features

- 🤖 **AI-Powered Classification**: Uses OpenAI GPT to intelligently classify notes
- 🔍 **Keyword Extraction**: Automatically extracts key topics from your notes
- 📊 **Similarity Matching**: Uses sentence embeddings to find similar existing notes
- 🎯 **Smart Suggestions**: Combines multiple methods for accurate categorization
- 📁 **Flexible Categories**: Easy to customize categories and subcategories

## How It Works

1. **Keyword Extraction**: Uses OpenAI to identify key terms and topics from your note
2. **AI Classification**: GPT analyzes the content and suggests the best category/subcategory
3. **Embedding Similarity**: Compares your note with existing categories using cosine similarity
4. **Final Suggestion**: Combines all methods to provide the best categorization result

## Setup

### Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. Clone or navigate to this repository:
```bash
cd MindEnhance
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or on Windows:
```bash
set OPENAI_API_KEY=your-api-key-here
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run main.py
```

2. Open your browser to the URL shown (typically `http://localhost:8501`)

3. Paste your note in the text box and click "🔍 Analyze & Classify"

4. View the suggested category (e.g., "This note fits under → Python > Decorators")

## Customizing Categories

Edit `categories.json` to add or modify categories and subcategories:

```json
{
  "categories": {
    "YourCategory": {
      "subcategories": ["Subcat1", "Subcat2"],
      "description": "Description of this category"
    }
  }
}
```

## Project Structure

```
MindEnhance/
├── main.py              # Streamlit app with classification logic
├── categories.json      # Category definitions
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Technologies Used

- **Streamlit**: Web interface
- **OpenAI API**: Text classification and keyword extraction
- **sentence-transformers**: Embeddings for similarity matching
- **NumPy**: Vector operations for cosine similarity

## Example Output

When you paste a note about Python decorators, you might see:

```
✅ This note fits under → Python > Decorators

Detailed Analysis:
- OpenAI Classification: Python > Decorators
- Embedding Similarity Match: Python > Decorators (similarity: 0.87)
- Extracted Keywords: decorator, function, wrapper, Python, syntax
```

## Notes

- The first run will download the sentence transformer model (~90MB)
- OpenAI API calls will incur costs (very minimal for this use case)
- Categories are stored in JSON and can be easily modified
