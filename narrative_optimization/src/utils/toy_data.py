"""
Toy data generation utilities for testing pipelines.

Uses sklearn's 20newsgroups dataset for initial validation before
moving to domain-specific data.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from pathlib import Path
import json


class ToyDataGenerator:
    """
    Generate and manage toy datasets for testing narrative pipelines.
    
    Uses 20newsgroups as the base dataset, with options for subsetting
    and validation against defined schemas.
    
    Parameters
    ----------
    data_dir : str, optional
        Directory for storing data
    random_state : int
        Random seed for reproducibility
    """
    
    def __init__(self, data_dir: str = "data/toy", random_state: int = 42):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        self.data_cache = {}
    
    def generate_20newsgroups_subset(
        self,
        n_categories: int = 4,
        n_samples_per_category: Optional[int] = None,
        categories: Optional[list] = None,
        subset: str = 'train'
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Generate a subset of 20newsgroups data.
        
        Parameters
        ----------
        n_categories : int
            Number of categories to include (if categories not specified)
        n_samples_per_category : int, optional
            Max samples per category (None = all samples)
        categories : list, optional
            Specific categories to use (None = auto-select)
        subset : str
            'train', 'test', or 'all'
        
        Returns
        -------
        X : array
            Text documents
        y : array
            Category labels
        target_names : list
            Category names
        """
        # Select categories
        if categories is None:
            all_categories = [
                'alt.atheism',
                'comp.graphics',
                'sci.space',
                'talk.religion.misc',
                'rec.sport.baseball',
                'sci.med',
                'comp.sys.mac.hardware',
                'talk.politics.misc'
            ]
            categories = all_categories[:n_categories]
        
        # Fetch data
        cache_key = f"{subset}_{len(categories)}"
        if cache_key in self.data_cache:
            newsgroups = self.data_cache[cache_key]
        else:
            newsgroups = fetch_20newsgroups(
                subset=subset,
                categories=categories,
                shuffle=True,
                random_state=self.random_state,
                remove=('headers', 'footers', 'quotes')
            )
            self.data_cache[cache_key] = newsgroups
        
        X = np.array(newsgroups.data)
        y = newsgroups.target
        target_names = newsgroups.target_names
        
        # Subsample if requested
        if n_samples_per_category is not None:
            indices = []
            for label in range(len(target_names)):
                label_indices = np.where(y == label)[0]
                selected = np.random.choice(
                    label_indices,
                    size=min(n_samples_per_category, len(label_indices)),
                    replace=False
                )
                indices.extend(selected)
            
            indices = np.array(indices)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
        
        return X, y, target_names
    
    def generate_train_test_split(
        self,
        n_train_samples: int = 400,
        n_test_samples: int = 100,
        n_categories: int = 4,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Generate complete train/test split for experiments.
        
        Parameters
        ----------
        n_train_samples : int
            Number of training samples
        n_test_samples : int
            Number of test samples
        n_categories : int
            Number of categories
        test_size : float
            Proportion for test split
        
        Returns
        -------
        data : dict
            Dictionary with X_train, X_test, y_train, y_test, target_names
        """
        # Get train data
        X_train, y_train, target_names = self.generate_20newsgroups_subset(
            n_categories=n_categories,
            n_samples_per_category=n_train_samples // n_categories,
            subset='train'
        )
        
        # Get test data
        X_test, y_test, _ = self.generate_20newsgroups_subset(
            n_categories=n_categories,
            n_samples_per_category=n_test_samples // n_categories,
            subset='test',
            categories=target_names
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'target_names': target_names,
            'metadata': {
                'n_train': len(X_train),
                'n_test': len(X_test),
                'n_categories': n_categories,
                'categories': target_names
            }
        }
    
    def save_dataset(self, data: Dict[str, Any], name: str):
        """
        Save dataset to disk.
        
        Parameters
        ----------
        data : dict
            Dataset dictionary
        name : str
            Dataset name
        """
        output_dir = self.data_dir / name
        output_dir.mkdir(exist_ok=True)
        
        # Save arrays
        np.save(output_dir / 'X_train.npy', data['X_train'])
        np.save(output_dir / 'X_test.npy', data['X_test'])
        np.save(output_dir / 'y_train.npy', data['y_train'])
        np.save(output_dir / 'y_test.npy', data['y_test'])
        
        # Save metadata
        metadata = {
            'target_names': data['target_names'],
            'metadata': data['metadata']
        }
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_dataset(self, name: str) -> Dict[str, Any]:
        """
        Load dataset from disk.
        
        Parameters
        ----------
        name : str
            Dataset name
        
        Returns
        -------
        data : dict
            Dataset dictionary
        """
        dataset_dir = self.data_dir / name
        
        if not dataset_dir.exists():
            raise ValueError(f"Dataset '{name}' not found in {self.data_dir}")
        
        # Load arrays
        X_train = np.load(dataset_dir / 'X_train.npy', allow_pickle=True)
        X_test = np.load(dataset_dir / 'X_test.npy', allow_pickle=True)
        y_train = np.load(dataset_dir / 'y_train.npy', allow_pickle=True)
        y_test = np.load(dataset_dir / 'y_test.npy', allow_pickle=True)
        
        # Load metadata
        with open(dataset_dir / 'metadata.json', 'r') as f:
            metadata_dict = json.load(f)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'target_names': metadata_dict['target_names'],
            'metadata': metadata_dict['metadata']
        }
    
    def validate_schema(self, data: Dict[str, Any]) -> bool:
        """
        Validate dataset against expected schema.
        
        Parameters
        ----------
        data : dict
            Dataset to validate
        
        Returns
        -------
        valid : bool
            Whether dataset is valid
        """
        required_keys = ['X_train', 'X_test', 'y_train', 'y_test', 'target_names', 'metadata']
        
        # Check required keys
        for key in required_keys:
            if key not in data:
                print(f"Missing required key: {key}")
                return False
        
        # Check shapes match
        if len(data['X_train']) != len(data['y_train']):
            print("X_train and y_train length mismatch")
            return False
        
        if len(data['X_test']) != len(data['y_test']):
            print("X_test and y_test length mismatch")
            return False
        
        # Check labels are valid
        n_categories = len(data['target_names'])
        if np.max(data['y_train']) >= n_categories or np.min(data['y_train']) < 0:
            print("Invalid y_train labels")
            return False
        
        if np.max(data['y_test']) >= n_categories or np.min(data['y_test']) < 0:
            print("Invalid y_test labels")
            return False
        
        return True
    
    def generate_and_save_default(self):
        """
        Generate and save the default toy dataset for experiments.
        
        Returns
        -------
        data : dict
            The generated dataset
        """
        print("Generating default toy dataset...")
        data = self.generate_train_test_split(
            n_train_samples=400,
            n_test_samples=100,
            n_categories=4
        )
        
        if self.validate_schema(data):
            print("Dataset validated successfully")
            self.save_dataset(data, 'default')
            print(f"Saved to {self.data_dir / 'default'}")
            
            # Print summary
            print(f"\nDataset Summary:")
            print(f"  Training samples: {len(data['X_train'])}")
            print(f"  Test samples: {len(data['X_test'])}")
            print(f"  Categories: {len(data['target_names'])}")
            print(f"  Category names: {data['target_names']}")
            
            return data
        else:
            raise ValueError("Generated dataset failed validation")


def quick_load_toy_data(data_dir: str = "data/toy") -> Dict[str, Any]:
    """
    Quick utility to load the default toy dataset.
    
    If it doesn't exist, generates it first.
    
    Parameters
    ----------
    data_dir : str
        Directory containing toy data
    
    Returns
    -------
    data : dict
        Dataset dictionary
    """
    generator = ToyDataGenerator(data_dir=data_dir)
    
    try:
        data = generator.load_dataset('default')
        print("Loaded existing toy dataset")
    except (ValueError, FileNotFoundError):
        print("Default dataset not found. Generating...")
        data = generator.generate_and_save_default()
    
    return data

