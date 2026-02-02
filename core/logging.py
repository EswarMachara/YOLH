# -*- coding: utf-8 -*-
"""
CSV Logging Module for RefYOLO-Human

Handles append-safe, reproducible CSV logging for training and evaluation.
No duplicated epochs. Deterministic.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
import sys

# File locking: fcntl on Unix, msvcrt on Windows
if sys.platform == 'win32':
    import msvcrt
    _HAS_FCNTL = False
else:
    import fcntl
    _HAS_FCNTL = True


class CSVLogger:
    """
    Append-safe CSV logger with no duplicate epochs.
    
    Usage:
        logger = CSVLogger("outputs/logs/train_metrics.csv")
        logger.log({"epoch": 1, "loss": 0.5, "acc": 0.8})
    """
    
    # Standard columns for grounding metrics
    # Note: rejection_accuracy removed - dataset contains no rejection samples
    GROUNDING_COLUMNS = [
        "epoch",
        "loss",
        "margin_success_rate",
        "accuracy_at_1",
        "mean_gt_rank",
        "pck_50",
        # "rejection_accuracy",  # Disabled: dataset contains no rejection samples
        "avg_gt_score",
        "avg_max_neg_score",
        "timestamp",
    ]
    
    # Standard columns for YOLO metrics
    YOLO_COLUMNS = [
        "epoch",
        "train_loss",
        "val_loss",
        "mAP50_box",
        "mAP50_mask",
        "OKS_pose",
        "timestamp",
    ]
    
    def __init__(
        self,
        filepath: Path,
        columns: Optional[List[str]] = None,
        overwrite: bool = False,
    ):
        """
        Initialize CSV logger.
        
        Args:
            filepath: Path to CSV file
            columns: Column names (if None, inferred from first log)
            overwrite: If True, overwrite existing file
        """
        self.filepath = Path(filepath)
        self.columns = columns
        self._initialized = False
        
        # Create parent directories
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Track logged epochs to prevent duplicates
        self._logged_epochs = set()
        
        # Handle existing file
        if self.filepath.exists() and not overwrite:
            self._load_existing()
        elif overwrite and self.filepath.exists():
            self.filepath.unlink()
    
    def _load_existing(self):
        """Load existing CSV to track logged epochs."""
        try:
            with open(self.filepath, 'r', newline='') as f:
                reader = csv.DictReader(f)
                self.columns = reader.fieldnames
                self._initialized = True
                
                for row in reader:
                    if 'epoch' in row and row['epoch']:
                        try:
                            self._logged_epochs.add(int(row['epoch']))
                        except (ValueError, TypeError):
                            pass
        except Exception as e:
            print(f"Warning: Could not load existing CSV: {e}")
    
    def log(self, data: Dict[str, Any], epoch: Optional[int] = None):
        """
        Log a row to CSV.
        
        Args:
            data: Dictionary of metric values
            epoch: Epoch number (optional, can be in data dict)
        
        Raises:
            ValueError: If epoch is already logged (prevents duplicates)
        """
        # Get epoch
        if epoch is None:
            epoch = data.get('epoch')
        else:
            data['epoch'] = epoch
        
        # Check for duplicate epoch
        if epoch is not None and epoch in self._logged_epochs:
            raise ValueError(f"Epoch {epoch} already logged. Duplicates not allowed.")
        
        # Add timestamp
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()
        
        # Initialize columns from data if not set
        if self.columns is None:
            self.columns = list(data.keys())
        
        # Write header if file doesn't exist
        write_header = not self.filepath.exists() or not self._initialized
        
        # Open file in append mode with file locking
        with open(self.filepath, 'a', newline='') as f:
            # File locking for concurrent safety
            if _HAS_FCNTL:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                except (OSError, IOError):
                    pass  # Fallback if locking not available
            elif sys.platform == 'win32':
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                except (OSError, IOError):
                    pass  # Fallback if locking not available
            
            writer = csv.DictWriter(f, fieldnames=self.columns, extrasaction='ignore')
            
            if write_header:
                writer.writeheader()
                self._initialized = True
            
            # Ensure all columns have values
            row = {col: data.get(col, '') for col in self.columns}
            writer.writerow(row)
        
        # Track logged epoch
        if epoch is not None:
            self._logged_epochs.add(epoch)
    
    def get_logged_epochs(self) -> set:
        """Get set of already logged epochs."""
        return self._logged_epochs.copy()
    
    def read_all(self) -> List[Dict[str, Any]]:
        """Read all rows from CSV."""
        if not self.filepath.exists():
            return []
        
        rows = []
        with open(self.filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric strings to numbers
                converted = {}
                for k, v in row.items():
                    try:
                        if '.' in str(v):
                            converted[k] = float(v)
                        else:
                            converted[k] = int(v)
                    except (ValueError, TypeError):
                        converted[k] = v
                rows.append(converted)
        return rows


class JSONLogger:
    """
    JSON logger for evaluation results.
    """
    
    def __init__(self, filepath: Path):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def save(self, data: Dict[str, Any]):
        """Save data to JSON file."""
        # Add metadata
        data['_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0',
        }
        
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load(self) -> Dict[str, Any]:
        """Load data from JSON file."""
        if not self.filepath.exists():
            return {}
        
        with open(self.filepath, 'r') as f:
            return json.load(f)


def create_train_logger(config) -> CSVLogger:
    """Create CSV logger for training metrics."""
    logs_dir = config.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    return CSVLogger(
        logs_dir / "train_metrics.csv",
        columns=CSVLogger.GROUNDING_COLUMNS,
    )


def create_val_logger(config) -> CSVLogger:
    """Create CSV logger for validation metrics."""
    logs_dir = config.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    return CSVLogger(
        logs_dir / "val_metrics.csv",
        columns=CSVLogger.GROUNDING_COLUMNS,
    )


def create_yolo_logger(config, model_type: str) -> CSVLogger:
    """Create CSV logger for YOLO training metrics."""
    logs_dir = config.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    return CSVLogger(
        logs_dir / f"yolo_{model_type}_metrics.csv",
        columns=CSVLogger.YOLO_COLUMNS,
    )
