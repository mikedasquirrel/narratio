"""
Data Loader

Unified data loading for all domain formats.

Author: Narrative Integration System
Date: November 2025
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple


class DataLoader:
    """
    Load data from multiple formats seamlessly.
    
    Supports:
    - JSON (multiple structures)
    - CSV
    - Pandas DataFrames
    - Direct arrays
    
    Handles:
    - Missing fields
    - Different naming conventions
    - Encoding issues
    - Empty values
    """
    
    def __init__(self):
        self.loaded_domains = {}
        
    def load(
        self,
        path: Union[Path, str],
        format: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Load data from file.
        
        Parameters
        ----------
        path : Path or str
            File path
        format : str, optional
            Force format ('json', 'csv', 'auto')
        
        Returns
        -------
        dict
            Loaded data with keys: texts, outcomes, names, timestamps
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        # Auto-detect format
        if format is None:
            format = path.suffix[1:]  # Remove dot
        
        if format == 'json':
            return self._load_json(path)
        elif format == 'csv':
            return self._load_csv(path)
        else:
            # Try JSON first, then CSV
            try:
                return self._load_json(path)
            except Exception:
                return self._load_csv(path)
    
    def _load_json(self, path: Path) -> Dict:
        """Load JSON file."""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
        
        # Handle different structures
        if isinstance(data, dict):
            texts = self._extract_field(data, ['texts', 'narratives', 'stories', 'descriptions'])
            outcomes = self._extract_field(data, ['outcomes', 'results', 'y', 'labels'])
            names = self._extract_field(data, ['names', 'entities', 'ids'])
            timestamps = self._extract_field(data, ['timestamps', 'dates', 'times'])
            
            # If no direct fields, check for list of records
            if not texts and 'data' in data:
                return self._parse_records(data['data'])
            elif not texts and 'records' in data:
                return self._parse_records(data['records'])
            
        elif isinstance(data, list):
            # List of records
            return self._parse_records(data)
        else:
            raise ValueError(f"Unknown JSON structure in {path}")
        
        # Convert to arrays
        outcomes = np.array(outcomes) if outcomes else np.zeros(len(texts))
        timestamps = np.array(timestamps) if timestamps else None
        
        return {
            'texts': texts,
            'outcomes': outcomes,
            'names': names,
            'timestamps': timestamps
        }
    
    def _load_csv(self, path: Path) -> Dict:
        """Load CSV file."""
        df = pd.read_csv(path, encoding='utf-8', errors='ignore')
        
        # Find text column
        text_col = self._find_column(df, ['narrative', 'text', 'description', 'story', 'content'])
        if not text_col:
            raise ValueError(f"No text column found in {path}")
        
        texts = df[text_col].fillna('').astype(str).tolist()
        
        # Find outcome column
        outcome_col = self._find_column(df, ['outcome', 'result', 'y', 'target', 'label', 'win'])
        outcomes = df[outcome_col].values if outcome_col else np.zeros(len(df))
        
        # Optional fields
        name_col = self._find_column(df, ['name', 'entity', 'id'])
        names = df[name_col].tolist() if name_col else None
        
        timestamp_col = self._find_column(df, ['timestamp', 'date', 'time', 'datetime'])
        timestamps = df[timestamp_col].values if timestamp_col else None
        
        return {
            'texts': texts,
            'outcomes': outcomes,
            'names': names,
            'timestamps': timestamps
        }
    
    def _extract_field(self, data: Dict, field_names: List[str]) -> Optional[List]:
        """Extract field trying multiple names."""
        for field_name in field_names:
            if field_name in data:
                value = data[field_name]
                if isinstance(value, list):
                    return value
                elif isinstance(value, np.ndarray):
                    return value.tolist()
        return None
    
    def _find_column(self, df: pd.DataFrame, col_names: List[str]) -> Optional[str]:
        """Find column trying multiple names."""
        for col_name in col_names:
            if col_name in df.columns:
                return col_name
        return None
    
    def _parse_records(self, records: List[Dict]) -> Dict:
        """Parse list of records."""
        texts = []
        outcomes = []
        names = []
        timestamps = []
        
        for record in records:
            # Extract text
            text = self._extract_field(record, ['narrative', 'text', 'description', 'story'])
            if text:
                texts.append(text[0] if isinstance(text, list) else text)
            else:
                texts.append(str(record))
            
            # Extract outcome
            outcome = self._extract_field(record, ['outcome', 'result', 'y', 'label'])
            outcomes.append(outcome[0] if isinstance(outcome, list) else (outcome if outcome is not None else 0))
            
            # Extract name
            name = self._extract_field(record, ['name', 'entity', 'id'])
            names.append(name[0] if isinstance(name, list) else name)
            
            # Extract timestamp
            timestamp = self._extract_field(record, ['timestamp', 'date', 'time'])
            timestamps.append(timestamp[0] if isinstance(timestamp, list) else timestamp)
        
        return {
            'texts': texts,
            'outcomes': np.array(outcomes),
            'names': names if any(n is not None for n in names) else None,
            'timestamps': np.array([t for t in timestamps if t is not None]) if any(t is not None for t in timestamps) else None
        }
    
    def validate_data(self, data: Dict) -> bool:
        """
        Validate loaded data.
        
        Parameters
        ----------
        data : dict
            Loaded data
        
        Returns
        -------
        bool
            True if valid
        """
        # Check required fields
        if 'texts' not in data or 'outcomes' not in data:
            return False
        
        texts = data['texts']
        outcomes = data['outcomes']
        
        # Check lengths match
        if len(texts) != len(outcomes):
            return False
        
        # Check not empty
        if len(texts) == 0:
            return False
        
        # Check for actual text content
        non_empty = sum(1 for t in texts if t and len(str(t).strip()) > 0)
        if non_empty < len(texts) * 0.5:  # At least 50% have content
            return False
        
        # Check outcome variance
        if len(np.unique(outcomes)) < 2:
            return False
        
        return True
    
    def load_domain(self, domain_name: str, data_path: Optional[Path] = None) -> Dict:
        """
        Load domain data with auto-discovery.
        
        Parameters
        ----------
        domain_name : str
            Domain name
        data_path : Path, optional
            Explicit path (if None, auto-discovers)
        
        Returns
        -------
        dict
            Loaded data
        """
        if data_path is None:
            # Auto-discover
            from ..pipeline_config import get_config
            config = get_config()
            data_path = config.get_domain_data_path(domain_name)
            
            if data_path is None:
                raise FileNotFoundError(f"No data file found for domain: {domain_name}")
        
        # Load
        data = self.load(data_path)
        
        # Validate
        if not self.validate_data(data):
            raise ValueError(f"Invalid data format for domain: {domain_name}")
        
        # Cache
        self.loaded_domains[domain_name] = data
        
        return data

