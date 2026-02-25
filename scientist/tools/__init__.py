"""
Scientist Tools Module
Contains various tools for scientific research and analysis

All tools inherit from the unified Tool base class and support optional LLM engine configuration.
"""

from .base_tool import Tool

# Import available tools
from .base_generator import Base_Generator_Tool
from .wikipedia_search import Wikipedia_Search_Tool
from .kegg_gene_search import KEGG_Gene_Search_Tool
from .pubmed_search import PubMed_Search_Tool
from .google_search import Google_Search_Tool
from .url_context_search import URL_Context_Search_Tool

# New tools
from .kegg_organism_search import KEGG_Organism_Search_Tool
from .kegg_drug_search import KEGG_Drug_Search_Tool
from .kegg_disease_search import KEGG_Disease_Search_Tool
from .perplexity_search import Perplexity_Search_Tool
from .mdipid_disease_search import MDIPID_Disease_Search_Tool
from .mdipid_microbe_search import MDIPID_Microbe_Search_Tool
from .mdipid_gene_search import MDIPID_Gene_Search_Tool
from .python_coder import Python_Coder_Tool

__all__ = [
    # Base class
    "Tool",

    # Tools
    "Base_Generator_Tool",
    "Wikipedia_Search_Tool",
    "KEGG_Gene_Search_Tool",
    "PubMed_Search_Tool",
    "Google_Search_Tool",
    "URL_Context_Search_Tool",
    "KEGG_Organism_Search_Tool",
    "KEGG_Drug_Search_Tool",
    "KEGG_Disease_Search_Tool",
    "Perplexity_Search_Tool",
    "MDIPID_Disease_Search_Tool",
    "MDIPID_Microbe_Search_Tool",
    "MDIPID_Gene_Search_Tool",
    "Python_Coder_Tool",
]

# Tool registry - useful for dynamic tool loading
TOOL_REGISTRY = {
    "Base_Generator_Tool": Base_Generator_Tool,
    "Wikipedia_Search_Tool": Wikipedia_Search_Tool,
    "KEGG_Gene_Search_Tool": KEGG_Gene_Search_Tool,
    "PubMed_Search_Tool": PubMed_Search_Tool,
    "Google_Search_Tool": Google_Search_Tool,
    "URL_Context_Search_Tool": URL_Context_Search_Tool,
    "KEGG_Organism_Search_Tool": KEGG_Organism_Search_Tool,
    "KEGG_Drug_Search_Tool": KEGG_Drug_Search_Tool,
    "KEGG_Disease_Search_Tool": KEGG_Disease_Search_Tool,
    "Perplexity_Search_Tool": Perplexity_Search_Tool,
    "MDIPID_Disease_Search_Tool": MDIPID_Disease_Search_Tool,
    "MDIPID_Microbe_Search_Tool": MDIPID_Microbe_Search_Tool,
    "MDIPID_Gene_Search_Tool": MDIPID_Gene_Search_Tool,
    "Python_Coder_Tool": Python_Coder_Tool,
}

# Tools that require LLM engine
TOOLS_REQUIRING_LLM = {
    name: tool_class
    for name, tool_class in TOOL_REGISTRY.items()
    if getattr(tool_class, 'require_llm_engine', False)
}

# Tools that don't require LLM engine
TOOLS_NO_LLM = {
    name: tool_class
    for name, tool_class in TOOL_REGISTRY.items()
    if not getattr(tool_class, 'require_llm_engine', False)
}
