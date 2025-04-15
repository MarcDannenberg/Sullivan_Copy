# sully_engine/codex.py
# 📚 Sully's Symbolic Codex (Knowledge Repository)

from datetime import datetime
import json
from typing import Dict, List, Any, Optional, Union, Set
import re
from collections import Counter

class SullyCodex:
    """
    Stores and organizes Sully's symbolic knowledge, concepts, and their relationships.
    Functions as both a lexicon and a semantic network of interconnected meanings.
    """

    def __init__(self):
        """Initialize an empty knowledge repository."""
        self.entries = {}
        self.terms = {}  # For word definitions
        self.associations = {}  # For tracking relationships between concepts

    def record(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Records new symbolic knowledge under a topic name.

        Args:
            topic: The symbolic topic or name
            data: Associated symbolic data or metadata
        """
        if not topic or not isinstance(data, dict):
            raise ValueError("Topic must be a non-empty string and data must be a dictionary")
            
        normalized_topic = topic.lower()
        self.entries[normalized_topic] = {
            **data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create associations with existing concepts
        self._create_associations(normalized_topic, data)

    def _create_associations(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Creates semantic associations between concepts based on shared attributes.
        
        Args:
            topic: The topic to create associations for
            data: The data containing potential association points
        """
        # Extract potential keywords from the data
        keywords = set()
        for value in data.values():
            if isinstance(value, str):
                # Split text into words, filter out very short words
                words = [w.lower() for w in str(value).split() if len(w) > 3]
                keywords.update(words)
        
        # Look for matches in existing entries
        for existing_topic in self.entries:
            if existing_topic == topic:
                continue  # Skip self-association
                
            # Check for keyword overlap in topic name
            if any(keyword in existing_topic for keyword in keywords):
                self._add_association(topic, existing_topic, "keyword_match")
                
            # Check for keyword overlap in data values
            existing_data = self.entries[existing_topic]
            existing_keywords = set()
            for value in existing_data.values():
                if isinstance(value, str):
                    words = [w.lower() for w in str(value).split() if len(w) > 3]
                    existing_keywords.update(words)
                    
            common_keywords = keywords.intersection(existing_keywords)
            if common_keywords:
                self._add_association(topic, existing_topic, "shared_concepts", list(common_keywords))

    def _add_association(self, topic1: str, topic2: str, type_: str, details: Any = None) -> None:
        """
        Adds a bidirectional association between two topics.
        
        Args:
            topic1: First topic
            topic2: Second topic
            type_: Type of association (e.g., "keyword_match", "shared_concepts")
            details: Optional details about the association
        """
        if not topic1 or not topic2 or not type_:
            return  # Skip invalid associations
            
        if topic1 not in self.associations:
            self.associations[topic1] = {}
            
        if topic2 not in self.associations:
            self.associations[topic2] = {}
            
        # Add bidirectional association
        self.associations[topic1][topic2] = {"type": type_, "details": details}
        self.associations[topic2][topic1] = {"type": type_, "details": details}

    def add_word(self, term: str, meaning: str) -> None:
        """
        Adds a new word definition to Sully's vocabulary.
        
        Args:
            term: The word or concept to define
            meaning: The definition or meaning of the term
        """
        if not term or not meaning:
            raise ValueError("Term and meaning must be non-empty strings")
            
        normalized_term = term.lower()
        self.terms[normalized_term] = {
            "meaning": meaning,
            "created": datetime.now().isoformat(),
            "contexts": []  # Tracks different contexts where the term appears
        }
        
        # Also add to entries for searchability
        self.record(normalized_term, {
            "type": "term",
            "definition": meaning
        })

    def add_context(self, term: str, context: str) -> None:
        """
        Adds a usage context for a term to enrich its understanding.
        
        Args:
            term: The term to add context for
            context: A sample sentence or context where the term is used
        """
        if not term or not context:
            return  # Skip empty terms or contexts
            
        normalized_term = term.lower()
        if normalized_term in self.terms:
            # Avoid duplicate contexts
            if context not in self.terms[normalized_term].get("contexts", []):
                self.terms[normalized_term].setdefault("contexts", []).append(context)
                # Update the timestamp
                self.terms[normalized_term]["updated"] = datetime.now().isoformat()

    def search(self, phrase: str, case_sensitive: bool = False, semantic: bool = True) -> Dict[str, Any]:
        """
        Searches the codex for entries matching a phrase, with optional
        semantic expansion to related concepts.

        Args:
            phrase: The search keyword
            case_sensitive: Match case when scanning
            semantic: Whether to include semantically related results

        Returns:
            Dictionary of matching entries (topic -> data)
        """
        if not phrase:
            return {}
            
        results = {}
        phrase_check = phrase if case_sensitive else phrase.lower()

        # Direct matches in entries
        for topic, data in self.entries.items():
            topic_check = topic if case_sensitive else topic.lower()
            
            # Check if phrase is in topic
            if phrase_check in topic_check:
                results[topic] = data
                continue
                
            # Check if phrase is in any value
            values = [str(v) for v in data.values() if v is not None]
            if any(phrase_check in (v if case_sensitive else v.lower()) for v in values):
                results[topic] = data

        # Search in term definitions
        for term, data in self.terms.items():
            term_check = term if case_sensitive else term.lower()
            meaning = data.get("meaning", "")
            meaning_check = meaning if case_sensitive else meaning.lower()
            
            if phrase_check in term_check or phrase_check in meaning_check:
                if term not in results:  # Avoid duplication with entries
                    results[term] = {
                        "type": "term",
                        "definition": meaning,
                        "contexts": data.get("contexts", [])
                    }

        # Expand to semantically related topics if requested
        if semantic and results:
            semantic_results = {}
            for topic in list(results.keys()):
                if topic in self.associations:
                    for related_topic, relation in self.associations[topic].items():
                        if related_topic not in results:
                            if related_topic in self.entries:
                                semantic_results[related_topic] = {
                                    **self.entries[related_topic],
                                    "related_to": topic,
                                    "relation": relation
                                }
            
            # Add semantic results with a note about their relationship
            results.update(semantic_results)

        return results

    def get(self, topic: str) -> Dict[str, Any]:
        """
        Gets a codex entry by topic name.
        
        Args:
            topic: The topic name to retrieve
            
        Returns:
            The entry data or a message if not found
        """
        if not topic:
            return {"message": "🔍 No topic specified."}
            
        normalized_topic = topic.lower()
        
        # Check entries first
        entry = self.entries.get(normalized_topic)
        if entry:
            # If it exists in entries, also check for associations
            result = dict(entry)
            if normalized_topic in self.associations:
                result["associations"] = {
                    related: info for related, info in self.associations[normalized_topic].items()
                }
            return result
            
        # Then check terms
        term_data = self.terms.get(normalized_topic)
        if term_data:
            return {
                "type": "term",
                "definition": term_data.get("meaning", ""),
                "contexts": term_data.get("contexts", []),
                "created": term_data.get("created"),
                "updated": term_data.get("updated")
            }
            
        return {"message": "🔍 No codex entry found."}

    def list_topics(self) -> List[str]:
        """
        Returns a list of all topic names currently in the codex.
        
        Returns:
            List of topic names
        """
        # Combine entries and terms (avoiding duplicates)
        all_topics = set(self.entries.keys())
        all_topics.update(self.terms.keys())
        return sorted(list(all_topics))

    def get_related_concepts(self, topic: str, max_depth: int = 1) -> Dict[str, Any]:
        """
        Gets concepts related to a given topic up to a specified depth of relationships.
        
        Args:
            topic: The topic to find related concepts for
            max_depth: How many relationship steps to traverse
            
        Returns:
            Dictionary of related concepts with their relationship paths
        """
        if not topic:
            return {}
            
        normalized_topic = topic.lower()
        if normalized_topic not in self.associations:
            return {}
            
        # Start with direct associations
        related = {
            related_topic: {"path": [normalized_topic], "relation": info}
            for related_topic, info in self.associations[normalized_topic].items()
        }
        
        # For depth > 1, traverse the graph further
        if max_depth > 1:
            current_level = list(related.keys())
            visited = set([normalized_topic])  # Track visited nodes to prevent cycles
            
            for depth in range(1, max_depth):
                next_level = []
                for current_topic in current_level:
                    if current_topic in self.associations:
                        for related_topic, info in self.associations[current_topic].items():
                            # Skip if already encountered to prevent cycles
                            if related_topic not in visited:
                                visited.add(related_topic)
                                path = related[current_topic]["path"] + [current_topic]
                                related[related_topic] = {
                                    "path": path,
                                    "relation": info,
                                    "depth": depth + 1
                                }
                                next_level.append(related_topic)
                                
                current_level = next_level
                if not current_level:
                    break  # No more connections to explore
                    
        return related

    def export(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns all codex entries for backup, JSON export, or UI rendering.
        Optionally saves to a file if path is provided.
        
        Args:
            path: Optional file path to save the export to
            
        Returns:
            Dictionary containing all codex data
        """
        export_data = {
            "entries": self.entries,
            "terms": self.terms,
            "associations": self.associations,
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "entry_count": len(self.entries),
                "term_count": len(self.terms),
                "association_count": sum(len(assocs) for assocs in self.associations.values())
            }
        }
        
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Error saving export: {str(e)}")
                
        return export_data

    def import_data(self, data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """
        Imports codex data from a previously exported format or file path.
        
        Args:
            data: Dictionary containing codex data or file path to JSON export
            
        Returns:
            Summary of imported data
        """
        if isinstance(data, str):
            # Treat as file path
            try:
                with open(data, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                return {"error": f"Failed to load file: {str(e)}"}
        
        if not isinstance(data, dict):
            return {"error": "Import data must be a dictionary or file path"}
            
        # Keep track of what was imported
        summary = {
            "entries_added": 0,
            "terms_added": 0,
            "associations_added": 0
        }
        
        # Import entries
        if "entries" in data and isinstance(data["entries"], dict):
            for topic, entry_data in data["entries"].items():
                if topic not in self.entries:
                    summary["entries_added"] += 1
                self.entries[topic] = entry_data
                
        # Import terms
        if "terms" in data and isinstance(data["terms"], dict):
            for term, term_data in data["terms"].items():
                if term not in self.terms:
                    summary["terms_added"] += 1
                self.terms[term] = term_data
                
        # Import associations
        if "associations" in data and isinstance(data["associations"], dict):
            for topic, assocs in data["associations"].items():
                if topic not in self.associations:
                    self.associations[topic] = {}
                    
                for related_topic, relation in assocs.items():
                    if related_topic not in self.associations.get(topic, {}):
                        summary["associations_added"] += 1
                    self.associations.setdefault(topic, {})[related_topic] = relation
        
        return summary

    def __len__(self) -> int:
        """
        Returns the total number of unique concepts in the codex.
        
        Returns:
            Count of unique concepts (entries + terms)
        """
        # Get unique set of all concepts (terms might overlap with entries)
        all_concepts = set(self.entries.keys())
        all_concepts.update(self.terms.keys())
        return len(all_concepts)
        
    def batch_process(self, text: str) -> List[Dict[str, Any]]:
        """
        Processes a text to extract and record potential concepts and their relationships.
        
        Args:
            text: Text to analyze for concepts
            
        Returns:
            List of newly identified and recorded concepts
        """
        if not text or not isinstance(text, str):
            return []
            
        # Extract potential concept phrases (sequences of 1-3 words)
        words = re.findall(r'\b[A-Za-z]{4,}\b', text)
        if not words:
            return []
            
        # Count word frequencies to identify important terms
        word_counts = Counter(words)
        important_words = [word for word, count in word_counts.most_common(10) if count > 1]
        
        # Record these as potential concepts
        new_concepts = []
        for word in important_words:
            # Extract a context for this word
            context_match = re.search(r'[^.!?]*\b' + re.escape(word) + r'\b[^.!?]*[.!?]', text)
            context = context_match.group(0).strip() if context_match else ""
            
            # Create a basic definition based on context
            definition = f"Concept extracted from text context: '{context}'"
            
            # Record in codex
            self.add_word(word, definition)
            if context:
                self.add_context(word, context)
                
            new_concepts.append({
                "term": word,
                "definition": definition,
                "context": context
            })
            
        return new_concepts

    def remove_entry(self, topic: str) -> bool:
        """
        Removes an entry from the codex and its associations.
        
        Args:
            topic: The topic to remove
            
        Returns:
            True if the entry was removed, False if not found
        """
        if not topic:
            return False
            
        normalized_topic = topic.lower()
        
        # Remove from entries
        entry_removed = False
        if normalized_topic in self.entries:
            del self.entries[normalized_topic]
            entry_removed = True
            
        # Remove from terms
        term_removed = False
        if normalized_topic in self.terms:
            del self.terms[normalized_topic]
            term_removed = True
            
        # Remove from associations
        if normalized_topic in self.associations:
            # Remove references to this topic from other topics' associations
            for other_topic in list(self.associations.keys()):
                if normalized_topic in self.associations[other_topic]:
                    del self.associations[other_topic][normalized_topic]
            
            # Remove the topic's associations
            del self.associations[normalized_topic]
        
        return entry_removed or term_removed

    def merge_topics(self, primary_topic: str, secondary_topic: str) -> bool:
        """
        Merges two topics, combining their data and associations.
        
        Args:
            primary_topic: The topic to keep
            secondary_topic: The topic to merge into the primary topic
            
        Returns:
            True if the merge was successful, False otherwise
        """
        if not primary_topic or not secondary_topic or primary_topic == secondary_topic:
            return False
            
        primary = primary_topic.lower()
        secondary = secondary_topic.lower()
        
        # Check if both topics exist
        if primary not in self.entries and secondary not in self.entries:
            return False
            
        # If primary doesn't exist but secondary does, swap them
        if secondary in self.entries and primary not in self.entries:
            primary, secondary = secondary, primary
            
        # Get secondary data
        secondary_data = self.entries.get(secondary, {})
        if not secondary_data:
            return False
            
        # Merge entries
        if primary in self.entries:
            # Update primary with non-overlapping fields from secondary
            for key, value in secondary_data.items():
                if key not in self.entries[primary] or key == "timestamp":
                    self.entries[primary][key] = value
        else:
            # Create primary from secondary
            self.entries[primary] = dict(secondary_data)
            
        # Merge term data if applicable
        if secondary in self.terms:
            if primary in self.terms:
                # Combine contexts
                self.terms[primary]["contexts"] = list(set(
                    self.terms[primary].get("contexts", []) + 
                    self.terms[secondary].get("contexts", [])
                ))
                # Keep the earliest created timestamp
                if "created" in self.terms[secondary]:
                    if ("created" not in self.terms[primary] or 
                        self.terms[secondary]["created"] < self.terms[primary]["created"]):
                        self.terms[primary]["created"] = self.terms[secondary]["created"]
            else:
                # Move secondary term data to primary
                self.terms[primary] = dict(self.terms[secondary])
                
        # Merge associations
        if secondary in self.associations:
            # Add all secondary associations to primary
            for related_topic, relation in self.associations[secondary].items():
                if related_topic != primary:  # Avoid self-association
                    self._add_association(primary, related_topic, relation["type"], relation.get("details"))
                    
            # Update other topics that were associated with secondary
            for other_topic in self.associations:
                if other_topic != primary and other_topic != secondary and secondary in self.associations[other_topic]:
                    relation = self.associations[other_topic][secondary]
                    self._add_association(other_topic, primary, relation["type"], relation.get("details"))
                    
        # Remove the secondary topic
        self.remove_entry(secondary)
        
        return True

    def statistics(self) -> Dict[str, Any]:
        """
        Returns statistical information about the codex contents.
        
        Returns:
            Dictionary with statistics about entries, terms, and associations
        """
        entry_count = len(self.entries)
        term_count = len(self.terms)
        
        # Get association metrics
        association_count = sum(len(assocs) for assocs in self.associations.values()) // 2  # Divided by 2 because associations are bidirectional
        
        # Find most connected topics
        topic_connections = {topic: len(assocs) for topic, assocs in self.associations.items()}
        most_connected = sorted(topic_connections.items(), key=lambda x: x[1], reverse=True)[:5] if topic_connections else []
        
        # Calculate average associations per topic
        avg_associations = association_count * 2 / len(self.associations) if self.associations else 0
        
        return {
            "total_concepts": len(self),
            "entries": entry_count,
            "terms": term_count,
            "associations": association_count,
            "avg_associations_per_topic": avg_associations,
            "most_connected_topics": most_connected
        }