"""
Notion API Integration Module
Handles all interactions with Notion for category management
Works with a page that contains a Table of Contents database
"""
import os
from typing import Dict, List, Optional
from notion_client import Client
import streamlit as st


class NotionCategoryManager:
    """Manages category operations with Notion API"""
    
    def __init__(self, api_key: Optional[str] = None, page_id: Optional[str] = None):
        """
        Initialize Notion client
        
        Args:
            api_key: Notion integration token (defaults to NOTION_API_KEY env var)
            page_id: Notion page ID that contains the Table of Contents database (defaults to NOTION_PAGE_ID env var)
        """
        self.api_key = api_key or os.getenv("NOTION_API_KEY")
        self.page_id = page_id or os.getenv("NOTION_PAGE_ID")
        
        if not self.api_key:
            raise ValueError("Notion API key is required. Set NOTION_API_KEY environment variable.")
        if not self.page_id:
            raise ValueError("Notion page ID is required. Set NOTION_PAGE_ID environment variable.")
        
        self.client = Client(auth=self.api_key)
        self.database_id = None  # Will be set when we find the database in the page
        self._find_database_in_page()
    
    def _query_database(self, database_id: str, filter_dict: Optional[Dict] = None) -> Dict:
        """
        Helper method to query a database, handling different API versions
        
        Args:
            database_id: The database ID to query
            filter_dict: Optional filter to apply to the query
        
        Returns:
            The response from the query
        """
        query_body = {}
        if filter_dict:
            query_body["filter"] = filter_dict
        
        try:
            # Try the standard notion-client API method
            if hasattr(self.client, 'databases') and hasattr(self.client.databases, 'query'):
                return self.client.databases.query(database_id=database_id, **query_body)
        except AttributeError:
            pass
        
        # Fallback: use requests library directly
        import requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }
        url = f"https://api.notion.com/v1/databases/{database_id}/query"
        response = requests.post(url, headers=headers, json=query_body)
        response.raise_for_status()
        return response.json()
    
    def _find_database_in_page(self):
        """Find the Table of Contents database within the page"""
        try:
            # Get all blocks from the page
            blocks_response = self.client.blocks.children.list(block_id=self.page_id)
            blocks = blocks_response.get("results", [])
            
            # Look for database blocks (child_database or database type)
            for block in blocks:
                block_type = block.get("type")
                
                # Check if it's a child database block
                if block_type == "child_database":
                    child_database = block.get("child_database", {})
                    if child_database:
                        self.database_id = block.get("id")
                        st.info(f"✅ Found Table of Contents database in page")
                        return
                
                # Also check for database blocks (if database is directly embedded)
                if block_type == "database":
                    self.database_id = block.get("id")
                    st.info(f"✅ Found Table of Contents database in page")
                    return
            
            # If not found in first level, check nested blocks (recursively)
            for block in blocks:
                if block.get("has_children", False):
                    nested_blocks = self._get_nested_blocks(block.get("id"))
                    for nested_block in nested_blocks:
                        nested_type = nested_block.get("type")
                        if nested_type == "child_database":
                            self.database_id = nested_block.get("id")
                            st.info(f"✅ Found Table of Contents database in page")
                            return
                        if nested_type == "database":
                            self.database_id = nested_block.get("id")
                            st.info(f"✅ Found Table of Contents database in page")
                            return
            
            # If still not found, raise an error
            raise ValueError(
                "No database found in the page. Please ensure:\n"
                "1. There is a Table of Contents database (inline or full page) in the page\n"
                "2. The page is shared with your Notion integration\n"
                "3. The integration has read access to the page"
            )
        
        except Exception as e:
            st.error(f"Error finding database in page: {e}")
            raise
    
    def _get_nested_blocks(self, block_id: str) -> List[Dict]:
        """Recursively get nested blocks"""
        try:
            blocks_response = self.client.blocks.children.list(block_id=block_id)
            blocks = blocks_response.get("results", [])
            
            all_blocks = []
            for block in blocks:
                all_blocks.append(block)
                if block.get("has_children", False):
                    nested = self._get_nested_blocks(block.get("id"))
                    all_blocks.extend(nested)
            
            return all_blocks
        except:
            return []
    
    def load_categories(self) -> Dict:
        """
        Load all categories from Notion database
        Subcategories are loaded from subpages within each category page
        
        Returns:
            Dict with structure: {"categories": {category_name: {subcategories: [], description: ""}}}
        """
        if not self.database_id:
            raise ValueError("Database ID not found. Cannot load categories.")
        
        try:
            # Query the database for all category pages
            response = self._query_database(self.database_id)
            
            categories = {}
            
            for page in response.get("results", []):
                # Extract category name from page title
                category_name = self._extract_title(page)
                if not category_name:
                    continue
                
                page_id = page.get("id")
                
                # Extract description
                description = self._extract_property(page, "Description", "rich_text") or ""
                
                # Extract subcategories from subpages
                subcategories = self._get_subpages_as_subcategories(page_id)
                
                categories[category_name] = {
                    "subcategories": subcategories,
                    "description": description
                }
            
            return {"categories": categories}
        
        except Exception as e:
            st.error(f"Error loading categories from Notion: {e}")
            raise
    
    def _get_subpages_as_subcategories(self, page_id: str) -> List[str]:
        """Get subcategories by finding subpages within a category page"""
        try:
            # Get child blocks from the page
            # Pages created via pages.create with parent page_id appear as child_page blocks
            blocks_response = self.client.blocks.children.list(block_id=page_id)
            blocks = blocks_response.get("results", [])
            
            subcategories = []
            
            for block in blocks:
                # Check if it's a child page (subpage)
                if block.get("type") == "child_page":
                    # Get the page ID from the block
                    child_page_id = block.get("id")
                    if child_page_id:
                        # Fetch the actual page to get its title
                        try:
                            child_page = self.client.pages.retrieve(page_id=child_page_id)
                            title = self._extract_title(child_page)
                            if title:
                                subcategories.append(title)
                        except:
                            # If we can't fetch the page, try to get title from block
                            child_page_data = block.get("child_page", {})
                            child_page_title = child_page_data.get("title", "")
                            if child_page_title:
                                subcategories.append(child_page_title)
            
            return subcategories
        except Exception as e:
            # If we can't get subpages, return empty list
            # This is non-fatal - categories can exist without subcategories
            return []
    
    def _extract_title(self, page: Dict) -> Optional[str]:
        """Extract title from page properties"""
        properties = page.get("properties", {})
        
        # Try common title property names
        for prop_name in ["Name", "Title", "Category", "Category Name"]:
            if prop_name in properties:
                prop = properties[prop_name]
                if prop.get("type") == "title":
                    title_array = prop.get("title", [])
                    if title_array:
                        return title_array[0].get("plain_text", "").strip()
        
        # Fallback: try to get any title property
        for prop_name, prop in properties.items():
            if prop.get("type") == "title":
                title_array = prop.get("title", [])
                if title_array:
                    return title_array[0].get("plain_text", "").strip()
        
        return None
    
    def _extract_property(self, page: Dict, property_name: str, property_type: str) -> Optional[str]:
        """Extract a property value from a page"""
        properties = page.get("properties", {})
        
        if property_name not in properties:
            return None
        
        prop = properties[property_name]
        if prop.get("type") != property_type:
            return None
        
        if property_type == "rich_text":
            rich_text_array = prop.get("rich_text", [])
            if rich_text_array:
                return rich_text_array[0].get("plain_text", "").strip()
        elif property_type == "title":
            title_array = prop.get("title", [])
            if title_array:
                return title_array[0].get("plain_text", "").strip()
        
        return None
    
    
    def add_category(self, category_name: str, description: str, subcategories: List[str]) -> bool:
        """
        Add a new category to Notion database
        
        Args:
            category_name: Name of the category
            description: Description of the category
            subcategories: List of subcategory names
        
        Returns:
            True if successful, False otherwise
        """
        if not self.database_id:
            st.error("Database ID not found. Cannot add category.")
            return False
        
        try:
            # Prepare properties for the new page
            properties = {}
            
            # Find the title property name
            database_info = self.client.databases.retrieve(database_id=self.database_id)
            title_prop_name = None
            
            for prop_name, prop in database_info.get("properties", {}).items():
                if prop.get("type") == "title":
                    title_prop_name = prop_name
                    break
            
            if not title_prop_name:
                # Try common names
                title_prop_name = "Name"
            
            # Set title
            properties[title_prop_name] = {
                "title": [{"text": {"content": category_name}}]
            }
            
            # Set description if property exists
            database_props = database_info.get("properties", {})
            desc_prop_name = None
            for prop_name in ["Description", "Desc"]:
                if prop_name in database_props and database_props[prop_name].get("type") == "rich_text":
                    desc_prop_name = prop_name
                    break
            
            if desc_prop_name:
                properties[desc_prop_name] = {
                    "rich_text": [{"text": {"content": description}}]
                }
            
            # Create the category page
            new_page = self.client.pages.create(
                parent={"database_id": self.database_id},
                properties=properties
            )
            
            # Create subpages for subcategories
            if subcategories:
                page_id = new_page.get("id")
                for subcat in subcategories:
                    self._create_subpage(page_id, subcat)
            
            return True
        
        except Exception as e:
            st.error(f"Error adding category to Notion: {e}")
            return False
    
    def _create_subpage(self, parent_page_id: str, subpage_title: str) -> bool:
        """Create a subpage (subcategory) within a parent page"""
        try:
            # Use pages.create with parent page_id to create a child page
            self.client.pages.create(
                parent={
                    "type": "page_id",
                    "page_id": parent_page_id
                },
                properties={
                    "title": {
                        "title": [
                            {
                                "text": {
                                    "content": subpage_title
                                }
                            }
                        ]
                    }
                }
            )
            return True
        except Exception as e:
            st.error(f"Error creating subpage '{subpage_title}': {e}")
            return False
    
    def add_subcategory(self, category_name: str, subcategory_name: str) -> bool:
        """
        Add a subcategory (subpage) to an existing category
        
        Args:
            category_name: Name of the existing category
            subcategory_name: Name of the new subcategory
        
        Returns:
            True if successful, False otherwise
        """
        if not self.database_id:
            st.error("Database ID not found. Cannot add subcategory.")
            return False
        
        try:
            # Find the category page
            title_prop_name = self._get_title_property_name()
            response = self._query_database(
                self.database_id,
                filter_dict={
                    "property": title_prop_name,
                    "title": {"equals": category_name}
                }
            )
            
            if not response.get("results"):
                st.error(f"Category '{category_name}' not found in Notion")
                return False
            
            page = response["results"][0]
            page_id = page["id"]
            
            # Check if subcategory already exists
            existing_subcategories = self._get_subpages_as_subcategories(page_id)
            if subcategory_name in existing_subcategories:
                return True  # Already exists
            
            # Create subpage
            return self._create_subpage(page_id, subcategory_name)
        
        except Exception as e:
            st.error(f"Error adding subcategory to Notion: {e}")
            return False
    
    def _get_title_property_name(self) -> str:
        """Get the title property name from the database"""
        if not self.database_id:
            return "Name"
        
        try:
            database_info = self.client.databases.retrieve(database_id=self.database_id)
            for prop_name, prop in database_info.get("properties", {}).items():
                if prop.get("type") == "title":
                    return prop_name
            return "Name"  # Default fallback
        except:
            return "Name"
    
    def update_category_description(self, category_name: str, description: str) -> bool:
        """Update the description of an existing category"""
        if not self.database_id:
            return False
        
        try:
            # Find the category page
            title_prop_name = self._get_title_property_name()
            response = self._query_database(
                self.database_id,
                filter_dict={
                    "property": title_prop_name,
                    "title": {"equals": category_name}
                }
            )
            
            if not response.get("results"):
                return False
            
            page = response["results"][0]
            page_id = page["id"]
            
            # Find description property
            database_info = self.client.databases.retrieve(database_id=self.database_id)
            desc_prop_name = None
            for prop_name in ["Description", "Desc"]:
                if prop_name in database_info.get("properties", {}):
                    prop = database_info["properties"][prop_name]
                    if prop.get("type") == "rich_text":
                        desc_prop_name = prop_name
                        break
            
            if not desc_prop_name:
                return False
            
            # Update the page
            self.client.pages.update(
                page_id=page_id,
                properties={
                    desc_prop_name: {
                        "rich_text": [{"text": {"content": description}}]
                    }
                }
            )
            
            return True
        
        except Exception as e:
            st.error(f"Error updating category description: {e}")
            return False
    
    def save_note_to_page(self, category_name: str, subcategory_name: Optional[str], note_text: str) -> bool:
        """
        Save a note text segment to the corresponding Notion page
        
        Args:
            category_name: Name of the category
            subcategory_name: Name of the subcategory (optional)
            note_text: The note text to save
        
        Returns:
            True if successful, False otherwise
        """
        if not self.database_id:
            return False
        
        try:
            # Find the category page
            title_prop_name = self._get_title_property_name()
            response = self._query_database(
                self.database_id,
                filter_dict={
                    "property": title_prop_name,
                    "title": {"equals": category_name}
                }
            )
            
            if not response.get("results"):
                return False
            
            category_page = response["results"][0]
            target_page_id = category_page.get("id")
            
            # If subcategory is specified, find or create the subcategory page
            if subcategory_name:
                # Check if subcategory page exists
                subcategories = self._get_subpages_as_subcategories(target_page_id)
                if subcategory_name in subcategories:
                    # Find the subcategory page ID
                    blocks_response = self.client.blocks.children.list(block_id=target_page_id)
                    blocks = blocks_response.get("results", [])
                    for block in blocks:
                        if block.get("type") == "child_page":
                            child_page_id = block.get("id")
                            try:
                                child_page = self.client.pages.retrieve(page_id=child_page_id)
                                if self._extract_title(child_page) == subcategory_name:
                                    target_page_id = child_page_id
                                    break
                            except:
                                continue
                else:
                    # Create subcategory page if it doesn't exist
                    if self._create_subpage(target_page_id, subcategory_name):
                        # Get the newly created page
                        blocks_response = self.client.blocks.children.list(block_id=target_page_id)
                        blocks = blocks_response.get("results", [])
                        for block in blocks:
                            if block.get("type") == "child_page":
                                child_page_id = block.get("id")
                                try:
                                    child_page = self.client.pages.retrieve(page_id=child_page_id)
                                    if self._extract_title(child_page) == subcategory_name:
                                        target_page_id = child_page_id
                                        break
                                except:
                                    continue
            
            # Add the note text to the page as a paragraph block
            self.client.blocks.children.append(
                block_id=target_page_id,
                children=[{
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {
                                    "content": note_text
                                }
                            }
                        ]
                    }
                }]
            )
            
            return True
        except Exception as e:
            st.error(f"Error saving note to Notion: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test the connection to Notion"""
        try:
            # Test by retrieving the page
            self.client.pages.retrieve(page_id=self.page_id)
            
            # Also test database access if we found it
            if self.database_id:
                self.client.databases.retrieve(database_id=self.database_id)
            
            return True
        except Exception as e:
            st.error(f"Notion connection test failed: {e}")
            return False
