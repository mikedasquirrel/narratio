"""
Oscar Winner Predictor - Live Model

Trained on 45 nominees, predicts winner probability for new films.
"""

import pickle
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.transformers.nominative import NominativeAnalysisTransformer
from src.transformers.emotional_resonance import EmotionalResonanceTransformer
from src.transformers.cultural_context import CulturalContextTransformer
from src.transformers.statistical import StatisticalTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class OscarPredictor:
    """Predicts Oscar win probability"""
    
    def __init__(self):
        self.transformers = None
        self.scaler = None
        self.model = None
        self.is_trained = False
    
    def train(self, films_data):
        """Train on Oscar data"""
        texts = [f['full_narrative'] for f in films_data]
        outcomes = np.array([f['won_oscar'] for f in films_data])
        
        # Setup transformers
        self.transformers = {
            'nominative': NominativeAnalysisTransformer(),
            'emotional': EmotionalResonanceTransformer(),
            'cultural': CulturalContextTransformer(),
            'statistical': StatisticalTransformer(max_features=100)
        }
        
        # Extract features
        all_features = []
        for name, trans in self.transformers.items():
            trans.fit(texts)
            feat = trans.transform(texts)
            if hasattr(feat, 'toarray'):
                feat = feat.toarray()
            all_features.append(feat)
        
        X = np.hstack(all_features)
        
        # Train model
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_scaled, outcomes)
        
        self.is_trained = True
        
        # Save model
        self.save_model()
    
    def predict(self, text, breakdown=True):
        """Predict win probability for new film"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Extract features
        all_features = []
        feature_contributions = {}
        
        for name, trans in self.transformers.items():
            feat = trans.transform([text])
            if hasattr(feat, 'toarray'):
                feat = feat.toarray()
            all_features.append(feat)
            
            if breakdown:
                # Get this transformer's contribution
                feat_scaled = self.scaler.transform(feat)
                contribution = np.mean(feat_scaled)
                feature_contributions[name] = float(contribution)
        
        X = np.hstack(all_features)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        win_prob = self.model.predict_proba(X_scaled)[0, 1]
        
        result = {
            'win_probability': float(win_prob),
            'prediction': 'WINNER' if win_prob > 0.5 else 'NOMINEE',
            'confidence': float(abs(win_prob - 0.5) * 2)
        }
        
        if breakdown:
            result['feature_contributions'] = feature_contributions
        
        return result
    
    def save_model(self):
        """Save trained model"""
        output_path = Path(__file__).parent / 'oscar_predictor_model.pkl'
        
        model_data = {
            'transformers': self.transformers,
            'scaler': self.scaler,
            'model': self.model
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved: {output_path}")
    
    def load_model(self):
        """Load trained model"""
        model_path = Path(__file__).parent / 'oscar_predictor_model.pkl'
        
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.transformers = model_data['transformers']
            self.scaler = model_data['scaler']
            self.model = model_data['model']
            self.is_trained = True
            
            return True
        return False


def train_model():
    """Train and save predictor"""
    from domains.oscars.data_loader import OscarDataLoader
    
    print("Training Oscar predictor...")
    loader = OscarDataLoader()
    films, _, _ = loader.load_full_dataset()
    
    predictor = OscarPredictor()
    predictor.train(films)
    
    # Test on training data
    correct = 0
    for film in films:
        pred = predictor.predict(film['full_narrative'], breakdown=False)
        if (pred['win_probability'] > 0.5 and film['won_oscar']) or \
           (pred['win_probability'] <= 0.5 and not film['won_oscar']):
            correct += 1
    
    accuracy = correct / len(films)
    print(f"\nTraining accuracy: {accuracy:.1%}")
    print("✓ Model ready for predictions!")


if __name__ == '__main__':
    train_model()

