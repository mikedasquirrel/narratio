"""
Input Validation Utility for Transformers

Centralized input validation to ensure all transformers receive proper string inputs.
Handles numpy arrays, bytes, pandas Series, and other edge cases.
"""

import numpy as np
from typing import List, Union, Any

# Optional pandas import
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


def ensure_string_list(X: Union[List, np.ndarray, Any]) -> List[str]:
    """
    Convert any input to a list of strings.
    
    Handles:
    - pandas Series
    - numpy arrays (including object dtype)
    - bytes objects
    - lists/tuples
    - single strings
    - mixed types
    
    Parameters
    ----------
    X : any
        Input data
        
    Returns
    -------
    list of str
        List of string documents
    """
    # Single string
    if isinstance(X, str):
        return [X]
    
    # Bytes
    if isinstance(X, bytes):
        return [X.decode('utf-8', errors='ignore')]
    
    # Pandas Series (CRITICAL FIX)
    if HAS_PANDAS and isinstance(X, pd.Series):
        return [str(item) for item in X.values]
    
    # Numpy array
    if isinstance(X, np.ndarray):
        if X.ndim == 0:
            # Scalar array
            return [str(X.item())]
        else:
            # Multi-dimensional - flatten and convert
            result = []
            for item in X.flat:
                if isinstance(item, bytes):
                    result.append(item.decode('utf-8', errors='ignore'))
                elif isinstance(item, np.str_) or isinstance(item, str):
                    result.append(str(item))
                elif isinstance(item, np.ndarray):
                    # Nested array
                    result.append(str(item))
                else:
                    result.append(str(item))
            return result
    
    # List or tuple
    if isinstance(X, (list, tuple)):
        result = []
        for item in X:
            if isinstance(item, bytes):
                result.append(item.decode('utf-8', errors='ignore'))
            elif isinstance(item, np.ndarray):
                # Handle nested arrays
                if item.ndim == 0:
                    result.append(str(item.item()))
                else:
                    result.append(str(item))
            elif isinstance(item, str):
                result.append(item)
            else:
                result.append(str(item))
        return result
    
    # Fallback
    return [str(X)]


def ensure_string(text: Any) -> str:
    """
    Convert any input to a string.
    
    Parameters
    ----------
    text : any
        Input text
        
    Returns
    -------
    str
        String text
    """
    if isinstance(text, bytes):
        return text.decode('utf-8', errors='ignore')
    elif isinstance(text, np.ndarray):
        if text.ndim == 0:
            return str(text.item())
        else:
            return str(text)
    elif isinstance(text, str):
        return text
    else:
        return str(text)

