import wikipedia
import logging
import requests
import sys
import re
from typing import List, Dict, Any, Optional, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ["search_wikipedia", "search_cellosaurus" ]


def safe_import(names: List[str]) -> Dict[str, Any]:
    """
    Safely import specified names from this module.
    
    Args:
        names: List of function names to import
        
    Returns:
        Dictionary mapping names to their corresponding functions
        
    Raises:
        ValueError: If a requested name doesn't exist or isn't allowed
    """
    # Define the allowed names that can be imported (already defined in __all__ above)
    allowed_names: Set[str] = set(__all__)
    
    # Validate all requested names
    for name in names:
        # Check if name contains only valid python identifier characters
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            raise ValueError(f"Invalid import name format: {name}")
        
        # Check if name is in the allowed list
        if name not in allowed_names:
            raise ValueError(f"Cannot import '{name}': not found or not allowed")
    
    # Import the current module
    current_module = sys.modules[__name__]
    
    # Build the dictionary of requested objects
    result = {}
    for name in names:
        result[name] = getattr(current_module, name)
    
    return result


def search_wikipedia(term:str):
    """
    A tool that searches wikipedia for a given page title, then fetches the page summary
    Args:
        term: A string representing the page title to search for
    """
    search_hits = wikipedia.search(term)
    if len(search_hits) == 0:
        logger.warning(f"No hits found for search term {term}, has the LLM done something dumb?")
        return "Your search returned no hits, try again. Remember, this will probably work best if you use a short identifier"
    
    try:
        summary = wikipedia.summary(search_hits[0])
    except:
        logger.error(f"Failed to pull a summary for the chosen page {search_hits[0]}")
        return "There isn't a good summary available, try a different search term"
    

    return summary


class CellosaurusAPI:
    """
    A Python wrapper for the Cellosaurus API to query cell line information.
    
    The Cellosaurus is a knowledge resource on cell lines with an emphasis on 
    providing information about what cell lines are used for and whether they
    are disease models or models for normal processes.
    """
    
    BASE_URL = "https://www.cellosaurus.org/api"
    
    def __init__(self):
        """Initialize the CellosaurusAPI wrapper."""
        # Get release information to verify API is accessible
        self._check_api_access()
    
    def _check_api_access(self) -> None:
        """
        Check if the API is accessible by getting release info.
        
        Raises:
            ConnectionError: If the API cannot be accessed
        """
        try:
            response = self.get_release_info()
            logger.info(f"Connected to Cellosaurus API (Version: {response.get('version', 'unknown')})")
        except Exception as e:
            raise ConnectionError(f"Could not connect to Cellosaurus API: {str(e)}")
    
    def get_release_info(self, format: str = "json") -> Dict[str, Any]:
        """
        Get information about the current Cellosaurus release.
        
        Args:
            format: Response format ('json', 'xml', 'txt', or 'tsv')
            
        Returns:
            Dictionary containing release information
        """
        endpoint = f"{self.BASE_URL}/release-info"
        params = {"format": format}
        
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        
        if format == "json":
            return response.json()
        else:
            return {"content": response.text}
    
    def get_cell_line(self, accession: str, fields: Optional[List[str]] = None, 
                      format: str = "json") -> Dict[str, Any]:
        """
        Get information about a specific cell line by its accession number.
        
        Args:
            accession: The cell line accession number (e.g., 'CVCL_S151')
            fields: Optional list of fields to return (e.g., ['id', 'ca', 'cc', 'ch'])
                   If None, all fields are returned
            format: Response format ('json', 'xml', 'txt', or 'tsv')
            
        Returns:
            Dictionary containing cell line information
        """
        endpoint = f"{self.BASE_URL}/cell-line/{accession}"
        
        params = {"format": format}
        if fields:
            params["fields"] = ",".join(fields)
        
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        
        if format == "json":
            return response.json()
        else:
            return {"content": response.text}
    
    def search_cell_lines(self, query: str = "id:HeLa", 
                          fields: Optional[List[str]] = None,
                          start: int = 0, 
                          rows: int = 10, 
                          sort: Optional[str] = None,
                          format: str = "json") -> Dict[str, Any]:
        """
        Search for cell lines using Solr query syntax.
        
        Args:
            query: Solr query string (e.g., 'id:HeLa', 'breast AND cancer')
            fields: Optional list of fields to return
            start: Index of first result to return
            rows: Number of results to return
            sort: Optional sort order (e.g., 'group asc,derived-from-site desc')
            format: Response format ('json', 'xml', 'txt', or 'tsv')
            
        Returns:
            Dictionary containing search results
        """
        endpoint = f"{self.BASE_URL}/search/cell-line"
        
        if not query.startswith("id:"):
            query = f"id:{query}"

        params = {
            "q": query,
            "start": start,
            "rows": rows,
            "format": format
        }
        
        if fields:
            params["fields"] = ",".join(fields)
        
        if sort:
            params["sort"] = sort
        
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        
        if format == "json":
            return response.json()
        else:
            return {"content": response.text}
    
    def is_disease_model(self, cell_line_data: Dict[str, Any]) -> bool:
        """
        Determine if a cell line is considered a disease model.
        
        Args:
            cell_line_data: Cell line data from get_cell_line or search_cell_lines
            
        Returns:
            True if the cell line is a disease model, False otherwise
        """
        # Check for disease-related keywords in various fields
        
        # Check caution field
        caution_field = cell_line_data.get("ca", [])
        for caution in caution_field:
            if any(keyword in caution.lower() for keyword in 
                  ["disease", "cancer", "tumor", "tumour", "patient", "patholog"]):
                return True
        
        # Check characteristics field
        characteristics_field = cell_line_data.get("ch", [])
        for characteristic in characteristics_field:
            if any(keyword in characteristic.lower() for keyword in 
                  ["disease", "cancer", "tumor", "tumour", "patient", "patholog", 
                   "malignant", "carcinoma", "leukemia", "disease model"]):
                return True
            
            # Specifically check for normal/healthy mentions that would contradict disease model
            if "normal" in characteristic.lower() or "healthy" in characteristic.lower():
                return False
        
        # Check comment field for disease indications
        comment_field = cell_line_data.get("cc", [])
        for comment in comment_field:
            if any(keyword in comment.lower() for keyword in 
                  ["disease model", "patient derived", "cancer model", "tumor model"]):
                return True
        
        # Default to False if no clear disease model indicators
        return False
    
    def get_cell_line_usage(self, cell_line_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract information about what the cell line is used for.
        
        Args:
            cell_line_data: Cell line data from get_cell_line or search_cell_lines
            
        Returns:
            Dictionary containing usage information
        """
        usage_info = {
            "is_disease_model": self.is_disease_model(cell_line_data),
            "applications": [],
            "cell_type": [],
            "derived_from_site": cell_line_data.get("derived-from-site", []),
            "characteristics": cell_line_data.get("ch", []),
            "comments": cell_line_data.get("cc", [])
        }
        
        # Extract cell type information
        cell_type_field = cell_line_data.get("cell-type", [])
        usage_info["cell_type"] = cell_type_field
        
        # Extract applications from comments
        comment_field = cell_line_data.get("cc", [])
        for comment in comment_field:
            if "used" in comment.lower() or "model" in comment.lower() or "suitable for" in comment.lower():
                usage_info["applications"].append(comment)
        
        return usage_info
    
def search_cellosaurus(cell_line_query:str) -> str:
    """
    A tool that searchers the cellosaurus API to get disease model and usage information for a given cell line

    Args:
        cell_line_query: A string representing the cell lines ID
    """
    api = CellosaurusAPI()
    
    # Get information about HeLa cell line
    search_results = api.search_cell_lines(cell_line_query)
    print(search_results['Cellosaurus']['cell-line-list'][0]['accession-list'][0]['value'])
    accession = search_results['Cellosaurus']['cell-line-list'][0]['accession-list'][0]['value']
    cell_info = api.get_cell_line(accession)
    
    # Get usage information
    usage = api.get_cell_line_usage(cell_info)
    if usage['is_disease_model']:
        summary = f"{cell_line_query} cell line is a disease model\n"
    else:
        summary = f"{cell_line_query} cell line is a normal process model\n"

    summary += "Applications:\n"
    for app in usage["applications"]:
        summary += f"- {app}"
    
    return summary