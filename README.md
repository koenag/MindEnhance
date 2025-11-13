# MindEnhance - Note Categorizer

An AI-powered note categorization tool that analyzes your notes and automatically suggests the best folder/category (like "Python > Decorators", "Career > Interview Prep", or "Machine Learning > Neural Networks") based on content.

## Features

- 🤖 **AI-Powered Classification**: Uses OpenAI GPT to intelligently classify notes
- 🔍 **Multi-Topic Detection**: Automatically splits notes with multiple distinct topics
- ✨ **Note Expansion**: AI can add context to make notes more meaningful
- 📊 **Similarity Matching**: Uses sentence embeddings to find similar existing notes
- 🎯 **Smart Suggestions**: Combines multiple methods for accurate categorization
- ➕ **Category Management**: Add new categories/subcategories directly from the UI
- 💾 **Auto-Save Categories**: Suggested new categories can be added with confirmation

## How It Works

1. **Multi-Topic Analysis**: AI detects if your note contains multiple distinct topics and splits them
2. **Note Expansion** (optional): AI adds context to make the note more self-contained
3. **Keyword Extraction**: Uses OpenAI to identify key terms and topics from your note
4. **AI Classification**: GPT analyzes each segment and suggests the best category/subcategory
5. **Embedding Similarity**: Compares your note with existing categories using cosine similarity
6. **Category Suggestions**: For new topics, AI suggests category names that you can add with confirmation

## Setup

### Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Notion account and workspace (optional - falls back to JSON if not configured)

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

### Notion Integration Setup (Recommended)

The app seamlessly integrates with Notion to manage your categories. If Notion is not configured, it will fall back to using a local `categories.json` file.

#### Step 1: Create a Notion Integration

1. Go to [Notion Integrations](https://www.notion.com/my-integrations)
2. Click "New integration"
3. Give it a name (e.g., "Note Categorizer")
4. Select your workspace
5. Under "Capabilities", enable:
   - **Read content**
   - **Update content**
   - **Insert content**
6. Click "Submit" and copy the **Internal Integration Token** (starts with `secret_`)

#### Step 2: Create a Page with Table of Contents Database

1. Create a new page in your Notion workspace (or use an existing page)
2. Inside the page, create a **Table of Contents** database:
   - Type `/database` or `/table` in the page
   - Select "Table - Inline" or "Table - Full page"
   - This will create a database block inside your page
3. Add the following properties to the database:
   - **Name** (Title) - This will be the category name
   - **Description** (Text/Rich Text) - Description of the category
4. Share the page with your integration:
   - Click the "..." menu in the top right of the page
   - Select "Connections" → "Add connections"
   - Search for and select your integration
   - Make sure the integration has access to the page

Database example: https://www.notion.so/2a93685360df803595c9d89f8952a40f?v=2a93685360df8031b1bc000c7a53de3e&source=copy_link

#### Step 3: Get Your Page ID

1. Open your Notion page (the one containing the Table of Contents database) in a web browser
2. Look at the URL - it will look like:
   ```
   https://www.notion.so/your-workspace/PAGE_ID?v=...
   ```
   Or if the page title is in the URL:
   ```
   https://www.notion.so/your-workspace/Page-Title-PAGE_ID
   ```
3. Copy the `PAGE_ID` (32-character string, may have hyphens)
   - The page ID is the part after the last hyphen in the URL

#### Step 4: Configure Environment Variables

Add to your `.env` file or export as environment variables:

```bash
export NOTION_API_KEY='secret_your_integration_token_here'
export NOTION_PAGE_ID='your-page-id-here'
```

Or create a `.env` file:
```
NOTION_API_KEY=secret_your_integration_token_here
NOTION_PAGE_ID=your-page-id-here
OPENAI_API_KEY=your-openai-key-here
```

**Important:** The app will automatically find the Table of Contents database within the page. You don't need to provide the database ID separately.

**Note:** The app will automatically fall back to `categories.json` if Notion is not configured, so you can use it without Notion if preferred.

## Usage

1. Run the Streamlit app:
```bash
streamlit run main.py
```

2. Open your browser to the URL shown (typically `http://localhost:8501`)

3. Paste your note in the text box

4. (Optional) Check "✨ Expand note" to have AI add context to your note

5. Click "🔍 Analyze & Classify"

6. View the suggested categories:
   - For single-topic notes: See the suggested category (e.g., "This note fits under → Python > Decorators")
   - For multi-topic notes: See each topic segment with its own category
   - For new categories: Confirm and add the suggested category with description and subcategories

7. Use the sidebar to manually add new categories or subcategories

## Customizing Categories

### Via UI (Recommended)
- **Add New Category**: Use the sidebar form to add categories with descriptions and subcategories
- **Add Subcategory**: Use the sidebar form to add subcategories to existing categories
- **Confirm AI Suggestions**: When AI suggests a new category, fill in the description and subcategories, then click "Add Category"

All changes are automatically saved to Notion (if configured) or to `categories.json` (fallback mode).

### Via Notion (If Integrated)
If you've set up Notion integration, you can:
- Edit categories directly in your Notion Table of Contents database (within the page)
- Changes will be reflected in the app (with a 60-second cache)
- The app will automatically sync with your Notion database
- The app automatically finds the database within your page - no need to specify the database ID

### Via JSON File (Fallback Mode)
If Notion is not configured, you can edit `categories.json` directly:

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
- **Notion API**: Category management and storage (with JSON fallback)
- **sentence-transformers**: Embeddings for similarity matching
- **NumPy**: Vector operations for cosine similarity

## Example Output

### Single Topic Note
When you paste a note about Python decorators, you might see:

```
✅ This segment fits under → Python > Decorators
Reasoning: The note discusses Python decorator syntax and usage patterns
Similarity Score: 0.87
```

### Multi-Topic Note
When you paste a note with multiple topics (e.g., "Python decorators are useful. Also, I had a great interview today."), you'll see:

```
🔀 Detected 2 distinct topics in your note

### Topic 1
✅ This segment fits under → Python > Decorators
📄 Relevant text segment: "Python decorators are useful..."

### Topic 2
✅ This segment fits under → Career > Interview Prep
📄 Relevant text segment: "I had a great interview today..."
```

### New Category Suggestion
When AI suggests a new category:

```
💡 Suggested new category: Quantum Computing
Reasoning: The note discusses quantum algorithms which don't fit existing categories

[Description input field]
[Subcategories input field]
[✅ Add Category button]
```

## Notes

- The first run will download the sentence transformer model (~90MB)
- OpenAI API calls will incur costs (very minimal for this use case)
- Categories are stored in Notion (if configured) or in `categories.json` (fallback)
- Notion integration is optional - the app works seamlessly with or without it
- Category data is cached for 60 seconds to reduce API calls
