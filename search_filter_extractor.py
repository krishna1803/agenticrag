"""Search filter extractor for generating predicates from natural language queries.

This module provides functionality to extract search filters and generate SQL predicates
for use in similarity search operations.
"""

import logging
from typing import Dict, List, Tuple, Optional
import re

logger = logging.getLogger(__name__)

# Mapping from filter equality operators to SQL operators
EQUALITY_LOOKUP = {"eq": "=", "ne": "!=", "gt": ">=", "lt": "<="}

class SearchFilterExtractor:
    """Extracts search filters and generates SQL predicates from natural language queries."""
    
    def __init__(self):
        """Initialize the search filter extractor."""
        pass
    
    def extract_simple_filters_from_query(self, query: str) -> List[Dict]:
        """Extract simple filters from a natural language query using pattern matching.
        
        This is a simplified version that extracts common legal document filters.
        
        Args:
            query: Natural language query
            
        Returns:
            List of filter dictionaries
        """
        filters = []
        query_lower = query.lower()
        
        # Extract year filters
        year_patterns = [
            r'(?:from|since|after)\s+(\d{4})',
            r'(?:before|prior to)\s+(\d{4})',
            r'in\s+(\d{4})',
            r'(\d{4})\s+(?:case|cases|decision|decisions)'
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, query_lower)
            for year in matches:
                filters.append({
                    "attribute": "date",
                    "equality": "eq" if "in" in pattern else ("gt" if any(word in pattern for word in ["from", "since", "after"]) else "lt"),
                    "value": year
                })
        
        # Extract document type filters
        if any(word in query_lower for word in ["case", "cases", "judgment", "judgments"]):
            filters.append({
                "attribute": "type",
                "equality": "eq",
                "value": "case"
            })
        elif any(word in query_lower for word in ["legislation", "act", "acts", "statute", "statutes"]):
            filters.append({
                "attribute": "type",
                "equality": "eq",
                "value": "legislation"
            })
        elif any(word in query_lower for word in ["journal", "article", "publication"]):
            filters.append({
                "attribute": "type",
                "equality": "eq",
                "value": "journal"
            })
        
        # Extract jurisdiction filters (ensure word boundaries to avoid false matches)
        jurisdictions = ["federal", "nsw", "vic", "qld", "wa", "sa", "tas", "nt", "act"]
        for jurisdiction in jurisdictions:
            # Use word boundaries to avoid matching "nt" in "contract"
            pattern = r'\b' + re.escape(jurisdiction) + r'\b'
            if re.search(pattern, query_lower):
                filters.append({
                    "attribute": "jurisdiction",
                    "equality": "eq",
                    "value": jurisdiction
                })
                break
        
        return filters

    def generate_search_predicate(self, search_filters: List[Dict]) -> Tuple[str, Dict]:
        """Generate a SQL predicate from the supplied filters compatible with PGVector cmetadata structure.
        
        Args:
            search_filters: List of filter dictionaries
            
        Returns:
            Tuple of (SQL predicate string, values dictionary)
        """
        predicate = []
        values = {}
        
        for att_filter in search_filters:
            # Ignore unknown equalities
            if att_filter.get("equality") not in EQUALITY_LOOKUP:
                logger.warning(
                    "Search filter contained an invalid equality operator [%s]",
                    att_filter.get("equality")
                )
                continue
                
            # Source and court have some special handling
            if att_filter.get("attribute") == "source" and att_filter.get("equality") == "eq":
                # Approximate match using 'ILIKE' - adapted for PGVector cmetadata structure
                source_name = att_filter.get("value", "").strip()
                year_matches = re.search(r"(?:\[|\()?(\d{4})(?:\]|\)?)$", source_name)
                if year_matches is not None:
                    source_name = source_name[:year_matches.start()].strip()
                predicate.append(
                    "(LOWER((cmetadata->>'name')) ILIKE %(source)s OR LOWER((cmetadata->>'title')) ILIKE %(source)s OR LOWER((cmetadata->>'titles')) ILIKE %(source)s)"
                )
                values["source"] = f"%{source_name.lower()}%"
                continue
                
            # TODO: Handle court appropriately when needed
            if att_filter.get("attribute") == "court":
                continue
                
            # Common legal document attributes for PGVector cmetadata structure
            filter_attributes = ["type", "jurisdiction", "date", "database", "name", "title"]
                
            if att_filter.get("attribute") in filter_attributes:
                attr_name = att_filter["attribute"]
                attr_value = att_filter.get("value", "").lower()
                equality_op = EQUALITY_LOOKUP[att_filter["equality"]]
                
                values[attr_name] = attr_value
                predicate.append(
                    f"LOWER((cmetadata->>'{attr_name}')) {equality_op} %({attr_name})s"
                )
        
        return " AND ".join(predicate), values
    
    def extract_predicate_from_query(self, query: str) -> Tuple[str, Dict]:
        """Extract filters from query and generate SQL predicate.
        
        Args:
            query: Natural language query
            
        Returns:
            Tuple of (SQL predicate string, values dictionary)
        """
        try:
            filters = self.extract_simple_filters_from_query(query)
            if not filters:
                logger.info(f"No filters extracted from query: {query}")
                return "", {}
                
            predicate, values = self.generate_search_predicate(filters)
            logger.info(f"Extracted predicate '{predicate}' with values {values} from query: {query}")
            return predicate, values
            
        except Exception as e:
            logger.error(f"Error extracting predicate from query '{query}': {str(e)}")
            return "", {}
