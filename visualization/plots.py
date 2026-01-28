# -*- coding: utf-8 -*-
"""
Plotting Utilities for RefYOLO-Human

Generates training/validation curves and metric visualizations.

FEATURES:
- ✅ Training vs Validation loss curves
- ✅ Margin success rate curves
- ✅ PCK@50 curves
- ✅ All metrics comparison
- ✅ Saves plots as PNG

USAGE:
    python visualization/plots.py --config config/config.yaml
    python visualization/plots.py --logs-dir outputs/logs --output-dir outputs/plots
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
from typing import List, Dict, Optional, Tuple
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


def load_csv_metrics(csv_path: Path) -> List[Dict]:
    """Load metrics from CSV file."""
    if not csv_path.exists():
        return []
    
    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
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


def plot_loss_curves(
    train_metrics: List[Dict],
    val_metrics: List[Dict],
    output_path: Path,
    title: str = "Training vs Validation Loss",
):
    """
    Plot training and validation loss curves.
    
    Args:
        train_metrics: List of training metric dicts
        val_metrics: List of validation metric dicts
        output_path: Path to save plot
        title: Plot title
    """
    if not HAS_MATPLOTLIB:
        print("Skipping plot: matplotlib not available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    train_epochs = [m['epoch'] for m in train_metrics if 'epoch' in m and 'loss' in m]
    train_losses = [m['loss'] for m in train_metrics if 'epoch' in m and 'loss' in m]
    
    val_epochs = [m['epoch'] for m in val_metrics if 'epoch' in m and 'loss' in m]
    val_losses = [m['loss'] for m in val_metrics if 'epoch' in m and 'loss' in m]
    
    # Plot
    if train_epochs:
        ax.plot(train_epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    if val_epochs:
        ax.plot(val_epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_margin_success_rate(
    train_metrics: List[Dict],
    val_metrics: List[Dict],
    output_path: Path,
    title: str = "Margin Success Rate",
):
    """Plot margin success rate curves."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    train_epochs = [m['epoch'] for m in train_metrics if 'margin_success_rate' in m]
    train_msr = [m['margin_success_rate'] * 100 for m in train_metrics if 'margin_success_rate' in m]
    
    val_epochs = [m['epoch'] for m in val_metrics if 'margin_success_rate' in m]
    val_msr = [m['margin_success_rate'] * 100 for m in val_metrics if 'margin_success_rate' in m]
    
    if train_epochs:
        ax.plot(train_epochs, train_msr, 'b-', label='Train', linewidth=2)
    if val_epochs:
        ax.plot(val_epochs, val_msr, 'r-', label='Val', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Margin Success Rate (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_pck_50(
    train_metrics: List[Dict],
    val_metrics: List[Dict],
    output_path: Path,
    title: str = "PCK@50",
):
    """Plot PCK@50 curves."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    train_epochs = [m['epoch'] for m in train_metrics if 'pck_50' in m]
    train_pck = [m['pck_50'] * 100 for m in train_metrics if 'pck_50' in m]
    
    val_epochs = [m['epoch'] for m in val_metrics if 'pck_50' in m]
    val_pck = [m['pck_50'] * 100 for m in val_metrics if 'pck_50' in m]
    
    if train_epochs:
        ax.plot(train_epochs, train_pck, 'b-', label='Train', linewidth=2)
    if val_epochs:
        ax.plot(val_epochs, val_pck, 'r-', label='Val', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('PCK@50 (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_accuracy_at_1(
    train_metrics: List[Dict],
    val_metrics: List[Dict],
    output_path: Path,
    title: str = "Accuracy@1",
):
    """Plot Accuracy@1 curves."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    train_epochs = [m['epoch'] for m in train_metrics if 'accuracy_at_1' in m]
    train_acc = [m['accuracy_at_1'] * 100 for m in train_metrics if 'accuracy_at_1' in m]
    
    val_epochs = [m['epoch'] for m in val_metrics if 'accuracy_at_1' in m]
    val_acc = [m['accuracy_at_1'] * 100 for m in val_metrics if 'accuracy_at_1' in m]
    
    if train_epochs:
        ax.plot(train_epochs, train_acc, 'b-', label='Train', linewidth=2)
    if val_epochs:
        ax.plot(val_epochs, val_acc, 'r-', label='Val', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy@1 (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_all_metrics_comparison(
    train_metrics: List[Dict],
    val_metrics: List[Dict],
    output_path: Path,
):
    """Create a multi-panel plot with all metrics."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    metrics_to_plot = [
        ('loss', 'Loss', False),
        ('margin_success_rate', 'Margin Success Rate (%)', True),
        ('accuracy_at_1', 'Accuracy@1 (%)', True),
        ('mean_gt_rank', 'Mean GT Rank', False),
        ('pck_50', 'PCK@50 (%)', True),
        ('avg_gt_score', 'Avg GT Score', False),
    ]
    
    for idx, (metric_key, metric_label, is_percentage) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        
        train_epochs = [m['epoch'] for m in train_metrics if metric_key in m]
        train_vals = [m[metric_key] for m in train_metrics if metric_key in m]
        
        val_epochs = [m['epoch'] for m in val_metrics if metric_key in m]
        val_vals = [m[metric_key] for m in val_metrics if metric_key in m]
        
        if is_percentage:
            train_vals = [v * 100 for v in train_vals]
            val_vals = [v * 100 for v in val_vals]
        
        if train_epochs:
            ax.plot(train_epochs, train_vals, 'b-', label='Train', linewidth=1.5)
        if val_epochs:
            ax.plot(val_epochs, val_vals, 'r-', label='Val', linewidth=1.5)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        if is_percentage:
            ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_yolo_metrics(
    pose_metrics: List[Dict],
    seg_metrics: List[Dict],
    output_path: Path,
):
    """Plot YOLO training metrics."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Train/Val Loss
    ax = axes[0]
    if pose_metrics:
        epochs = [m['epoch'] for m in pose_metrics if 'train_loss' in m]
        train_loss = [float(m['train_loss']) if m['train_loss'] else 0 for m in pose_metrics if 'train_loss' in m]
        val_loss = [float(m['val_loss']) if m['val_loss'] else 0 for m in pose_metrics if 'val_loss' in m]
        
        ax.plot(epochs, train_loss, 'b-', label='Pose Train', linewidth=1.5)
        ax.plot(epochs, val_loss, 'b--', label='Pose Val', linewidth=1.5)
    
    if seg_metrics:
        epochs = [m['epoch'] for m in seg_metrics if 'train_loss' in m]
        train_loss = [float(m['train_loss']) if m['train_loss'] else 0 for m in seg_metrics if 'train_loss' in m]
        val_loss = [float(m['val_loss']) if m['val_loss'] else 0 for m in seg_metrics if 'val_loss' in m]
        
        ax.plot(epochs, train_loss, 'r-', label='Seg Train', linewidth=1.5)
        ax.plot(epochs, val_loss, 'r--', label='Seg Val', linewidth=1.5)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('YOLO Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # mAP50
    ax = axes[1]
    if pose_metrics:
        epochs = [m['epoch'] for m in pose_metrics if 'mAP50_box' in m]
        map50 = [float(m['mAP50_box']) if m['mAP50_box'] else 0 for m in pose_metrics if 'mAP50_box' in m]
        ax.plot(epochs, map50, 'b-', label='Pose mAP50(B)', linewidth=1.5)
    
    if seg_metrics:
        epochs = [m['epoch'] for m in seg_metrics if 'mAP50_box' in m]
        map50_b = [float(m['mAP50_box']) if m['mAP50_box'] else 0 for m in seg_metrics if 'mAP50_box' in m]
        map50_m = [float(m['mAP50_mask']) if m['mAP50_mask'] else 0 for m in seg_metrics if 'mAP50_mask' in m]
        
        ax.plot(epochs, map50_b, 'r-', label='Seg mAP50(B)', linewidth=1.5)
        ax.plot(epochs, map50_m, 'r--', label='Seg mAP50(M)', linewidth=1.5)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP50')
    ax.set_title('YOLO mAP50')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # OKS (pose only)
    ax = axes[2]
    if pose_metrics:
        epochs = [m['epoch'] for m in pose_metrics if 'OKS_pose' in m]
        oks = [float(m['OKS_pose']) if m['OKS_pose'] else 0 for m in pose_metrics if 'OKS_pose' in m]
        ax.plot(epochs, oks, 'b-', label='Pose OKS', linewidth=1.5)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('OKS')
    ax.set_title('YOLO Pose OKS')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def generate_all_plots(
    logs_dir: Path,
    output_dir: Path,
):
    """
    Generate all plots from log files.
    
    Args:
        logs_dir: Directory containing CSV log files
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating plots from: {logs_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load grounding metrics
    train_csv = logs_dir / "train_metrics.csv"
    val_csv = logs_dir / "val_metrics.csv"
    
    train_metrics = load_csv_metrics(train_csv)
    val_metrics = load_csv_metrics(val_csv)
    
    if train_metrics or val_metrics:
        print(f"\nGrounding metrics:")
        print(f"  Train epochs: {len(train_metrics)}")
        print(f"  Val epochs: {len(val_metrics)}")
        
        # Generate grounding plots
        plot_loss_curves(train_metrics, val_metrics, output_dir / "loss_curves.png")
        plot_margin_success_rate(train_metrics, val_metrics, output_dir / "margin_success_rate.png")
        plot_pck_50(train_metrics, val_metrics, output_dir / "pck_50.png")
        plot_accuracy_at_1(train_metrics, val_metrics, output_dir / "accuracy_at_1.png")
        plot_all_metrics_comparison(train_metrics, val_metrics, output_dir / "all_metrics.png")
    else:
        print("\nNo grounding metrics found")
    
    # Load YOLO metrics
    pose_csv = logs_dir / "yolo_pose_metrics.csv"
    seg_csv = logs_dir / "yolo_seg_metrics.csv"
    
    pose_metrics = load_csv_metrics(pose_csv)
    seg_metrics = load_csv_metrics(seg_csv)
    
    if pose_metrics or seg_metrics:
        print(f"\nYOLO metrics:")
        print(f"  Pose epochs: {len(pose_metrics)}")
        print(f"  Seg epochs: {len(seg_metrics)}")
        
        plot_yolo_metrics(pose_metrics, seg_metrics, output_dir / "yolo_metrics.png")
    else:
        print("\nNo YOLO metrics found")
    
    print(f"\n✓ All plots generated in: {output_dir}")


def main():
    """Command-line interface for plotting."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate training plots")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--logs-dir", type=str, default=None, help="Logs directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib required for plotting")
        print("Install with: pip install matplotlib")
        sys.exit(1)
    
    print("=" * 60)
    print("GENERATING TRAINING PLOTS")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine directories
    if args.config:
        from core.config import load_config
        config = load_config(args.config)
        logs_dir = config.logs_dir
        output_dir = config.plots_dir
    else:
        logs_dir = Path(args.logs_dir or "outputs/logs")
        output_dir = Path(args.output_dir or "outputs/plots")
    
    generate_all_plots(logs_dir, output_dir)


if __name__ == "__main__":
    main()
