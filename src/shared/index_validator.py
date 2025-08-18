#!/usr/bin/env python3
"""
Index Validator Utility for ECC Data Generation Pipeline

Validates semantic indexes for consistency, completeness, and functionality.
Performs comprehensive checks on FAISS indexes, models, and data quality.

Author: Marleen de Jonge
Date: 2025
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from ..analysis_pipeline.base_searcher import BaseSemanticSearcher


class IndexValidator:
    """
    Utility class for validating semantic indexes.
    Performs comprehensive validation checks on loaded indexes.
    """
    
    def __init__(self, searcher: 'BaseSemanticSearcher'):
        self.searcher = searcher
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the loaded index for consistency and completeness.
        
        Returns:
            Tuple of (is_valid: bool, issues: List[str])
            - is_valid: True if index passes all validation checks
            - issues: List of validation issues found (empty if valid)
            
        Raises:
            RuntimeError: If index is not loaded
        """
        if not getattr(self.searcher, '_index_loaded', False):
            raise RuntimeError("Index not loaded. Call load_index() first.")
        
        issues = []
        
        try:
            self.logger.info("🔍 Validating index integrity...")
            
            # Run all validation checks
            issues.extend(self._validate_component_consistency())
            issues.extend(self._validate_model_compatibility())
            issues.extend(self._validate_data_quality())
            issues.extend(self._validate_faiss_functionality())
            issues.extend(self._validate_configuration())
            
            # Set validation status
            is_valid = len(issues) == 0
            self.searcher._index_validated = is_valid
            self.searcher._validation_date = datetime.now().isoformat()
            
            # Log results
            if is_valid:
                self.logger.info("✅ Index validation passed - no issues found")
            else:
                self.logger.warning(f"⚠️ Index validation found {len(issues)} issues:")
                for i, issue in enumerate(issues, 1):
                    self.logger.warning(f"   {i}. {issue}")
            
            return is_valid, issues
            
        except Exception as e:
            error_msg = f"Index validation failed with error: {e}"
            self.logger.error(error_msg)
            return False, [error_msg]
    
    def _validate_component_consistency(self) -> List[str]:
        """Check consistency between FAISS index, snippets, and metadata."""
        issues = []
        
        snippets_count = len(self.searcher.snippets)
        faiss_count = self.searcher.faiss_index.ntotal
        metadata_count = self.searcher.metadata.get('num_snippets', 0)
        
        if snippets_count != faiss_count:
            issues.append(
                f"Snippet count ({snippets_count}) doesn't match "
                f"FAISS vectors ({faiss_count})"
            )
        
        if snippets_count != metadata_count:
            issues.append(
                f"Snippet count ({snippets_count}) doesn't match "
                f"metadata count ({metadata_count})"
            )
        
        # Check embedding dimensions
        expected_dim = self.searcher.metadata.get('embedding_dimension')
        actual_dim = self.searcher.faiss_index.d
        
        if expected_dim and actual_dim != expected_dim:
            issues.append(
                f"FAISS dimension ({actual_dim}) doesn't match "
                f"metadata dimension ({expected_dim})"
            )
        
        return issues
    
    def _validate_model_compatibility(self) -> List[str]:
        """Check model and embedding compatibility."""
        issues = []
        
        if not self.searcher.model:
            issues.append("No model loaded")
            return issues
        
        try:
            # Test encoding to get actual dimension
            test_embedding = self.searcher.model.encode(["test"], normalize_embeddings=True)
            model_dim = test_embedding.shape[1]
            index_dim = self.searcher.faiss_index.d
            
            if model_dim != index_dim:
                issues.append(
                    f"Model embedding dimension ({model_dim}) doesn't match "
                    f"FAISS index dimension ({index_dim})"
                )
        except Exception as e:
            issues.append(f"Model encoding test failed: {e}")
        
        return issues
    
    def _validate_data_quality(self) -> List[str]:
        """Check data quality issues in snippets."""
        issues = []
        
        # Check for empty texts
        empty_texts = sum(1 for s in self.searcher.snippets 
                         if not s.text or not s.text.strip())
        if empty_texts > 0:
            issues.append(f"Found {empty_texts} snippets with empty text")
        
        # Check for missing core metadata
        missing_metadata = sum(1 for s in self.searcher.snippets 
                              if not s.company_name or not s.ticker or not s.date)
        if missing_metadata > 0:
            issues.append(f"Found {missing_metadata} snippets with missing core metadata")
        
        # Check date validity
        invalid_dates = 0
        for snippet in self.searcher.snippets:
            try:
                if snippet.date:
                    pd.to_datetime(snippet.date)
            except:
                invalid_dates += 1
        
        if invalid_dates > 0:
            issues.append(f"Found {invalid_dates} snippets with invalid dates")
        
        # Check for potential duplicates (sample-based for performance)
        duplicate_issues = self._check_duplicates()
        if duplicate_issues:
            issues.extend(duplicate_issues)
        
        return issues
    
    def _check_duplicates(self) -> List[str]:
        """Check for duplicate snippets (optimized for large datasets)."""
        issues = []
        
        if len(self.searcher.snippets) > 1000:
            # Sample-based duplicate check for large indexes
            sample_size = min(1000, len(self.searcher.snippets))
            sample_indices = np.random.choice(len(self.searcher.snippets), sample_size, replace=False)
            sample_texts = [self.searcher.snippets[i].text[:100] for i in sample_indices]
            if len(set(sample_texts)) < len(sample_texts):
                issues.append("Potential duplicate snippets detected (sample-based check)")
        else:
            # Full duplicate check for smaller indexes
            texts = [s.text[:100] for s in self.searcher.snippets]
            unique_texts = len(set(texts))
            total_texts = len(texts)
            
            if unique_texts < total_texts:
                duplicates = total_texts - unique_texts
                issues.append(f"Found {duplicates} potential duplicate snippets")
        
        return issues
    
    def _validate_faiss_functionality(self) -> List[str]:
        """Test FAISS index search functionality."""
        issues = []
        
        if len(self.searcher.snippets) == 0:
            issues.append("No snippets available for FAISS functionality test")
            return issues
        
        try:
            # Test search with the first snippet
            test_text = self.searcher.snippets[0].text
            test_embedding = self.searcher.model.encode([test_text], normalize_embeddings=True)
            scores, indices = self.searcher.faiss_index.search(test_embedding.astype(np.float32), 5)
            
            if len(scores[0]) == 0:
                issues.append("FAISS search returned no results for test query")
            elif indices[0][0] < 0 or indices[0][0] >= len(self.searcher.snippets):
                issues.append("FAISS search returned invalid indices")
            elif scores[0][0] < 0.5:  # Should be high similarity to itself
                issues.append("FAISS search returned unexpectedly low similarity for identical text")
                
        except Exception as e:
            issues.append(f"FAISS search functionality test failed: {e}")
        
        return issues
    
    def _validate_configuration(self) -> List[str]:
        """Validate configuration if available."""
        issues = []
        
        if not hasattr(self.searcher, 'config') or not self.searcher.config:
            # No configuration is not an error, just note it
            return issues
        
        try:
            # Test configuration loading
            if hasattr(self.searcher.config, 'get_available_analyses'):
                available_analyses = self.searcher.config.get_available_analyses()
                if not available_analyses:
                    issues.append("Configuration loaded but no analyses available")
            else:
                issues.append("Configuration loaded but missing expected methods")
                
        except Exception as e:
            issues.append(f"Configuration validation failed: {e}")
        
        return issues