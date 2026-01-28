# Evaluation module for RefYOLO-Human
from .grounding_eval import evaluate_test_split, main as run_evaluation

__all__ = [
    'evaluate_test_split',
    'run_evaluation',
]
