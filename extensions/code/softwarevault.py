"""
Software Vault for storing and retrieving code snippets.

This module provides a simple vault for storing code snippets with
metadata and searching them based on various criteria.
"""

import time
from typing import List, Dict, Optional
import re

class SoftwareVault:
    def __init__(self):
        """Initialize an empty software vault."""
        self.library = []

    def store_snippet(self, name: str, language: str, description: str, code: str, source: Optional[str] = None) -> Dict:
        """
        Store a code snippet with metadata in the vault.
        
        Args:
            name: A descriptive name for the snippet
            language: The programming language of the snippet
            description: A detailed description of what the snippet does
            code: The actual code
            source: Optional source reference (URL, book, etc.)
            
        Returns:
            The stored snippet entry
        """
        snippet = {
            "id": len(self.library) + 1,
            "name": name,
            "language": language,
            "description": description,
            "code": code,
            "source": source,
            "timestamp": time.time()
        }
        self.library.append(snippet)
        return snippet

    def search_by_name(self, query: str) -> List[Dict]:
        """
        Search snippets by name.
        
        Args:
            query: The search term to look for in snippet names
            
        Returns:
            A list of matching snippets
        """
        return [item for item in self.library if query.lower() in item["name"].lower()]

    def search_by_language(self, language: str) -> List[Dict]:
        """
        Search snippets by programming language.
        
        Args:
            language: The programming language to filter by
            
        Returns:
            A list of matching snippets
        """
        return [item for item in self.library if language.lower() == item["language"].lower()]

    def search_by_text(self, query: str) -> List[Dict]:
        """
        Search snippets by any text field (name, description, code).
        
        Args:
            query: The search term to look for in any text field
            
        Returns:
            A list of matching snippets
        """
        results = []
        query_lower = query.lower()
        
        for item in self.library:
            if (query_lower in item["name"].lower() or
                query_lower in item["description"].lower() or
                query_lower in item["code"].lower()):
                results.append(item)
                
        return results

    def get_best_for_goal(self, goal: str) -> List[Dict]:
        """
        Find snippets that best match a specified goal.
        
        Args:
            goal: A description of what the user wants to accomplish
            
        Returns:
            Up to 5 snippets that best match the goal
        """
        # Split goal into keywords for better matching
        keywords = re.findall(r'\w+', goal.lower())
        
        # Score each snippet based on keyword matches
        scored_snippets = []
        for item in self.library:
            score = 0
            desc_lower = item["description"].lower()
            
            # Score based on keyword presence in description
            for keyword in keywords:
                if keyword in desc_lower:
                    score += 1
            
            # Give extra points for exact phrase match
            if goal.lower() in desc_lower:
                score += 5
                
            if score > 0:
                scored_snippets.append((score, item))
        
        # Sort by score (descending) and return top 5
        sorted_snippets = [item for _, item in sorted(scored_snippets, key=lambda x: x[0], reverse=True)]
        return sorted_snippets[:5]

    def get_all(self) -> List[Dict]:
        """
        Get all snippets in the vault.
        
        Returns:
            A list of all snippets
        """
        return self.library
    
    def delete_snippet(self, snippet_id: int) -> bool:
        """
        Delete a snippet by its ID.
        
        Args:
            snippet_id: The ID of the snippet to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        for i, item in enumerate(self.library):
            if item.get("id") == snippet_id:
                self.library.pop(i)
                return True
        return False

# Create a global instance for easy access
vault = SoftwareVault()