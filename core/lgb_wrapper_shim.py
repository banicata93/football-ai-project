"""
LGBWrapper Shim for Backward Compatibility

This module provides a minimal shim class to enable unpickling of 
Draw Specialist v1 model that was trained with a custom LGBWrapper class.

The shim allows the model to be loaded without breaking existing functionality.
It does NOT affect any other models or pipelines.

Usage: Import this module before loading Draw Specialist pickle files.
"""


class LGBWrapper:
    """
    Minimal shim class for unpickling Draw Specialist v1 model.
    
    This class was used during training but not properly saved in a module.
    The shim allows backward compatibility without retraining.
    """
    
    def __init__(self, model=None):
        """
        Initialize wrapper with optional model.
        
        Args:
            model: LightGBM model instance (optional)
        """
        self.model = model
        self._booster = None
        
        # If model has _Booster attribute, extract it
        if hasattr(model, '_Booster'):
            self._booster = model._Booster
        elif hasattr(model, 'booster_'):
            self._booster = model.booster_
    
    def predict(self, X):
        """Predict class labels"""
        if self.model is not None:
            if hasattr(self.model, 'predict'):
                return self.model.predict(X)
        return None
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if self.model is not None:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
        return None
    
    def __getstate__(self):
        """Support pickling"""
        return self.__dict__
    
    def __setstate__(self, state):
        """Support unpickling"""
        self.__dict__.update(state)
