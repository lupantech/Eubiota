"""
Global Embedding Store - Singleton for preloading and sharing embeddings across all tools.

This module provides a centralized, thread-safe store for embeddings used by KEGG and MDIPID tools.
Embeddings are loaded once at application startup and shared across all tool instances,
regardless of which LLM model they use.

Benefits:
- Memory efficiency: Each embedding is loaded only once
- Faster initialization: Tools don't need to load embeddings on every instantiation
- Model independence: Embeddings are shared across gpt-4o, claude, etc.

Usage:
    from scientist.utils.embedding_store import GlobalEmbeddingStore
    
    # At application startup (e.g., in main.py)
    store = GlobalEmbeddingStore.get_instance()
    store.preload_all("/path/to/project")
    
    # In tool __init__
    store = GlobalEmbeddingStore.get_instance()
    self.embeddings = store.get_embeddings("kegg_disease_database")
"""

import os
import json
import pickle as pkl
import threading
from typing import Dict, Any, Optional
import time


class GlobalEmbeddingStore:
    """
    Singleton class for managing globally shared embeddings.
    
    Thread-safe and supports preloading at application startup.
    """
    
    _instance: Optional['GlobalEmbeddingStore'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if hasattr(self, '_init_done') and self._init_done:
            return
        
        self._embeddings: Dict[str, Any] = {}
        self._gene_data: Dict[str, Any] = {}  # For MDIPID gene data (JSON files)
        self._gene_id_list: list = []
        self._preload_lock = threading.Lock()
        self._initialized = False
        self._preload_stats: Dict[str, Any] = {}
        self._init_done = True
    
    @classmethod
    def get_instance(cls) -> 'GlobalEmbeddingStore':
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @property
    def is_initialized(self) -> bool:
        """Check if embeddings have been preloaded."""
        return self._initialized
    
    def preload_all(self, project_root: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Preload all embeddings from KEGG and MDIPID databases.
        
        This should be called once at application startup.
        Thread-safe: multiple calls will only load once.
        
        Args:
            project_root: Path to the project root directory
            verbose: Whether to print loading progress
            
        Returns:
            Dict with loading statistics
        """
        if self._initialized:
            return self._preload_stats
        
        with self._preload_lock:
            if self._initialized:
                return self._preload_stats
            
            start_time = time.time()
            stats = {
                "loaded_databases": [],
                "failed_databases": [],
                "memory_estimate_mb": 0,
                "load_time_seconds": 0
            }
            
            embedding_model = "text-embedding-3-large"
            
            # Define all databases to preload
            databases = [
                # KEGG databases
                ("kegg_disease_database", "kegg"),
                ("kegg_drug_database", "kegg"),
                ("kegg_gene_database", "kegg"),
                ("kegg_organism_database", "kegg"),
                # MDIPID databases (with embeddings)
                ("mdipid_disease_database", "mdipid"),
                ("mdipid_microbiota_database", "mdipid"),
            ]
            
            for db_name, db_type in databases:
                db_path = os.path.join(project_root, "data", db_name)
                embedding_path = os.path.join(db_path, "embeddings", f"{embedding_model}.pkl")
                
                if os.path.exists(embedding_path):
                    try:
                        file_size = os.path.getsize(embedding_path) / (1024 * 1024)  # MB
                        
                        with open(embedding_path, "rb") as f:
                            self._embeddings[db_name] = pkl.load(f)
                        
                        record_count = len(self._embeddings[db_name])
                        stats["loaded_databases"].append({
                            "name": db_name,
                            "records": record_count,
                            "file_size_mb": round(file_size, 2)
                        })
                        stats["memory_estimate_mb"] += file_size
                        
                        if verbose:
                            print(f"✓ Preloaded {db_name}: {record_count} records ({file_size:.1f} MB)")
                            
                    except Exception as e:
                        stats["failed_databases"].append({
                            "name": db_name,
                            "error": str(e)
                        })
                        if verbose:
                            print(f"✗ Failed to load {db_name}: {e}")
                else:
                    stats["failed_databases"].append({
                        "name": db_name,
                        "error": f"File not found: {embedding_path}"
                    })
                    if verbose:
                        print(f"! Skipped {db_name}: embeddings file not found")
            
            # Special handling for MDIPID Gene database (JSON files, not pkl)
            gene_db_path = os.path.join(project_root, "data", "mdipid_gene_database")
            gene_id_list_path = os.path.join(gene_db_path, "gene_id_list.json")
            
            if os.path.exists(gene_id_list_path):
                try:
                    with open(gene_id_list_path, 'r', encoding='utf-8') as f:
                        self._gene_id_list = json.load(f)
                    
                    # Load all gene data files
                    gene_raw_dir = os.path.join(gene_db_path, "raw")
                    loaded_genes = 0
                    total_size = 0
                    
                    for gene_id in self._gene_id_list:
                        gene_file_path = os.path.join(gene_raw_dir, f"{gene_id}.json")
                        if os.path.exists(gene_file_path):
                            total_size += os.path.getsize(gene_file_path)
                            with open(gene_file_path, 'r', encoding='utf-8') as f:
                                self._gene_data[gene_id] = json.load(f)
                            loaded_genes += 1
                    
                    size_mb = total_size / (1024 * 1024)
                    stats["loaded_databases"].append({
                        "name": "mdipid_gene_database",
                        "records": loaded_genes,
                        "file_size_mb": round(size_mb, 2)
                    })
                    stats["memory_estimate_mb"] += size_mb
                    
                    if verbose:
                        print(f"✓ Preloaded mdipid_gene_database: {loaded_genes} genes ({size_mb:.1f} MB)")
                        
                except Exception as e:
                    stats["failed_databases"].append({
                        "name": "mdipid_gene_database",
                        "error": str(e)
                    })
                    if verbose:
                        print(f"✗ Failed to load mdipid_gene_database: {e}")
            
            stats["load_time_seconds"] = round(time.time() - start_time, 2)
            stats["file_size_mb"] = round(stats["memory_estimate_mb"], 2)  # Renamed for clarity
            # Actual memory is ~4.5x larger due to Python object overhead
            stats["memory_estimate_mb"] = round(stats["memory_estimate_mb"] * 4.5, 2)
            
            self._preload_stats = stats
            self._initialized = True
            
            if verbose:
                print(f"\n=== Embedding Store Initialized ===")
                print(f"Total databases: {len(stats['loaded_databases'])}")
                print(f"File size on disk: {stats['file_size_mb']:.1f} MB")
                print(f"Estimated memory usage: {stats['memory_estimate_mb']:.0f} MB (~{stats['memory_estimate_mb']/1024:.1f} GB)")
                print(f"Load time: {stats['load_time_seconds']:.2f}s")
            
            return stats
    
    def get_embeddings(self, database_name: str) -> Dict[str, Any]:
        """
        Get preloaded embeddings for a specific database.
        
        Args:
            database_name: Name of the database (e.g., "kegg_disease_database")
            
        Returns:
            Dict containing the embeddings, or empty dict if not found
        """
        return self._embeddings.get(database_name, {})
    
    def get_gene_data(self) -> Dict[str, Any]:
        """
        Get preloaded MDIPID gene data.
        
        Returns:
            Dict mapping gene_id to gene data
        """
        return self._gene_data
    
    def get_gene_id_list(self) -> list:
        """
        Get preloaded MDIPID gene ID list.
        
        Returns:
            List of gene IDs
        """
        return self._gene_id_list
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loading statistics."""
        return self._preload_stats
    
    def clear(self):
        """Clear all cached embeddings (for testing or memory recovery)."""
        with self._preload_lock:
            self._embeddings.clear()
            self._gene_data.clear()
            self._gene_id_list.clear()
            self._initialized = False
            self._preload_stats = {}


# Convenience function for preloading
def preload_embeddings(project_root: str = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function to preload all embeddings.
    
    Args:
        project_root: Path to project root. If None, auto-detect from this file's location.
        verbose: Whether to print loading progress
        
    Returns:
        Loading statistics
    """
    if project_root is None:
        # Auto-detect project root from this file's location
        current_file = os.path.abspath(__file__)
        # scientist/utils/embedding_store.py -> project_root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    
    store = GlobalEmbeddingStore.get_instance()
    return store.preload_all(project_root, verbose=verbose)


if __name__ == "__main__":
    # Test the embedding store
    print("Testing GlobalEmbeddingStore...")
    
    store = GlobalEmbeddingStore.get_instance()
    stats = store.preload_all(project_root="/home/ubuntu/science_agent_dev")
    
    print("\n--- Stats ---")
    import json
    print(json.dumps(stats, indent=2))
    
    # Test getting embeddings
    kegg_disease = store.get_embeddings("kegg_disease_database")
    print(f"\nKEGG Disease records: {len(kegg_disease)}")
    
    mdipid_microbe = store.get_embeddings("mdipid_microbiota_database")
    print(f"MDIPID Microbe records: {len(mdipid_microbe)}")
    
    gene_ids = store.get_gene_id_list()
    print(f"MDIPID Gene IDs: {len(gene_ids)}")

