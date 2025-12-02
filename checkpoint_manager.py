#!/usr/bin/env python3
"""
Checkpoint Management Utilities for Nested Learning

This script provides utilities to manage, inspect, and compare checkpoints
from Nested Learning chat sessions.
"""

import torch
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import sys

class CheckpointManager:
    """Utilities for managing Nested Learning checkpoints"""
    
    def __init__(self, checkpoint_dir: str = "./nested_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def list_checkpoints(self, verbose: bool = False) -> List[Dict]:
        """List all available checkpoints"""
        checkpoints = []
        
        for checkpoint_path in sorted(self.checkpoint_dir.iterdir()):
            if not checkpoint_path.is_dir():
                continue
            
            metadata_path = checkpoint_path / "metadata.json"
            if not metadata_path.exists():
                continue
            
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Get file sizes
                weights_path = checkpoint_path / "fast_weights.pt"
                weights_size = weights_path.stat().st_size if weights_path.exists() else 0
                
                checkpoint_info = {
                    "name": checkpoint_path.name,
                    "path": str(checkpoint_path),
                    "session_id": metadata.get("session_id", "unknown"),
                    "timestamp": metadata.get("timestamp", "unknown"),
                    "turn_count": metadata.get("turn_count", 0),
                    "total_adaptations": metadata.get("total_adaptations", 0),
                    "conversation_length": metadata.get("conversation_length", 0),
                    "weights_size_mb": weights_size / (1024 * 1024),
                    "config": metadata.get("config", {}),
                }
                
                checkpoints.append(checkpoint_info)
                
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not read {checkpoint_path}: {e}")
                continue
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        return checkpoints
    
    def print_checkpoint_info(self, checkpoint_name: str):
        """Print detailed information about a checkpoint"""
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found: {checkpoint_name}")
            return
        
        # Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print("\n" + "="*70)
        print(f"CHECKPOINT: {checkpoint_name}")
        print("="*70)
        print(f"Session ID: {metadata.get('session_id', 'N/A')}")
        print(f"Timestamp: {metadata.get('timestamp', 'N/A')}")
        print(f"Turn Count: {metadata.get('turn_count', 0)}")
        print(f"Total Adaptations: {metadata.get('total_adaptations', 0)}")
        print(f"Conversation Length: {metadata.get('conversation_length', 0)} messages")
        
        print("\nConfiguration:")
        config = metadata.get('config', {})
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Weights info
        weights_path = checkpoint_path / "fast_weights.pt"
        if weights_path.exists():
            weights = torch.load(weights_path, map_location='cpu')
            print(f"\nFast Weights:")
            print(f"  Number of tensors: {len(weights)}")
            total_params = sum(t.numel() for t in weights.values())
            print(f"  Total parameters: {total_params:,}")
            print(f"  File size: {weights_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Conversation history
        history_path = checkpoint_path / "conversation.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
            print(f"\nConversation History:")
            print(f"  Messages: {len(history)}")
            if history:
                print(f"  First message: {history[0][1][:50]}...")
                print(f"  Last message: {history[-1][1][:50]}...")
        
        print("="*70 + "\n")
    
    def compare_checkpoints(self, checkpoint1: str, checkpoint2: str):
        """Compare two checkpoints"""
        path1 = self.checkpoint_dir / checkpoint1
        path2 = self.checkpoint_dir / checkpoint2
        
        if not path1.exists() or not path2.exists():
            print("Error: One or both checkpoints not found")
            return
        
        # Load weights
        weights1 = torch.load(path1 / "fast_weights.pt", map_location='cpu')
        weights2 = torch.load(path2 / "fast_weights.pt", map_location='cpu')
        
        print("\n" + "="*70)
        print(f"COMPARING: {checkpoint1} vs {checkpoint2}")
        print("="*70)
        
        # Check if same parameters
        if set(weights1.keys()) != set(weights2.keys()):
            print("⚠ Warning: Checkpoints have different parameter sets!")
            print(f"  Only in {checkpoint1}: {set(weights1.keys()) - set(weights2.keys())}")
            print(f"  Only in {checkpoint2}: {set(weights2.keys()) - set(weights1.keys())}")
        
        # Compute differences
        common_keys = set(weights1.keys()) & set(weights2.keys())
        total_diff = 0
        max_diff = 0
        max_diff_param = None
        
        for key in common_keys:
            diff = torch.abs(weights1[key] - weights2[key]).sum().item()
            total_diff += diff
            param_max_diff = torch.abs(weights1[key] - weights2[key]).max().item()
            if param_max_diff > max_diff:
                max_diff = param_max_diff
                max_diff_param = key
        
        print(f"\nParameter Differences:")
        print(f"  Common parameters: {len(common_keys)}")
        print(f"  Total absolute difference: {total_diff:.6e}")
        print(f"  Max difference: {max_diff:.6e} (in {max_diff_param})")
        print("="*70 + "\n")
    
    def export_weights(self, checkpoint_name: str, output_path: str):
        """Export checkpoint weights to a standalone file"""
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found: {checkpoint_name}")
            return
        
        weights_path = checkpoint_path / "fast_weights.pt"
        weights = torch.load(weights_path, map_location='cpu')
        
        # Save with metadata
        export_data = {
            "weights": weights,
            "checkpoint_name": checkpoint_name,
            "export_timestamp": datetime.now().isoformat(),
        }
        
        # Load original metadata
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            export_data["original_metadata"] = json.load(f)
        
        torch.save(export_data, output_path)
        print(f"✓ Exported checkpoint to: {output_path}")
    
    def cleanup_old_checkpoints(self, keep_recent: int = 5, dry_run: bool = True):
        """Remove old checkpoints, keeping only the N most recent"""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_recent:
            print(f"No cleanup needed. Found {len(checkpoints)} checkpoints, keeping {keep_recent}.")
            return
        
        to_delete = checkpoints[keep_recent:]
        
        print(f"\n{'DRY RUN - ' if dry_run else ''}Cleanup Plan:")
        print(f"  Total checkpoints: {len(checkpoints)}")
        print(f"  Keeping: {keep_recent} most recent")
        print(f"  Deleting: {len(to_delete)} old checkpoints")
        print()
        
        for ckpt in to_delete:
            print(f"  - {ckpt['name']} (from {ckpt['timestamp']})")
            if not dry_run:
                import shutil
                shutil.rmtree(ckpt['path'])
        
        if dry_run:
            print("\nRun with --no-dry-run to actually delete files.")
        else:
            print(f"\n✓ Deleted {len(to_delete)} old checkpoints")


def main():
    parser = argparse.ArgumentParser(
        description="Manage Nested Learning checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all checkpoints
  python checkpoint_manager.py list
  
  # Show detailed info about a checkpoint
  python checkpoint_manager.py info checkpoint_20240101_120000_turn10
  
  # Compare two checkpoints
  python checkpoint_manager.py compare ckpt1 ckpt2
  
  # Export checkpoint to standalone file
  python checkpoint_manager.py export checkpoint_name output.pt
  
  # Clean up old checkpoints (keep 5 most recent)
  python checkpoint_manager.py cleanup --keep 5
        """
    )
    
    parser.add_argument(
        'command',
        choices=['list', 'info', 'compare', 'export', 'cleanup'],
        help='Command to execute'
    )
    
    parser.add_argument(
        'args',
        nargs='*',
        help='Command arguments'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        default='./nested_checkpoints',
        help='Checkpoint directory (default: ./nested_checkpoints)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--keep',
        type=int,
        default=5,
        help='Number of checkpoints to keep (for cleanup command)'
    )
    
    parser.add_argument(
        '--no-dry-run',
        action='store_true',
        help='Actually perform cleanup (default is dry run)'
    )
    
    args = parser.parse_args()
    
    manager = CheckpointManager(args.checkpoint_dir)
    
    if args.command == 'list':
        checkpoints = manager.list_checkpoints(verbose=args.verbose)
        
        if not checkpoints:
            print("No checkpoints found.")
            return
        
        print("\n" + "="*70)
        print("AVAILABLE CHECKPOINTS")
        print("="*70)
        
        for i, ckpt in enumerate(checkpoints, 1):
            print(f"\n{i}. {ckpt['name']}")
            print(f"   Session: {ckpt['session_id']}")
            print(f"   Timestamp: {ckpt['timestamp']}")
            print(f"   Turns: {ckpt['turn_count']}, Adaptations: {ckpt['total_adaptations']}")
            print(f"   Conversation: {ckpt['conversation_length']} messages")
            print(f"   Size: {ckpt['weights_size_mb']:.2f} MB")
        
        print("="*70 + "\n")
    
    elif args.command == 'info':
        if not args.args:
            print("Error: Checkpoint name required")
            print("Usage: python checkpoint_manager.py info <checkpoint_name>")
            return
        
        manager.print_checkpoint_info(args.args[0])
    
    elif args.command == 'compare':
        if len(args.args) < 2:
            print("Error: Two checkpoint names required")
            print("Usage: python checkpoint_manager.py compare <checkpoint1> <checkpoint2>")
            return
        
        manager.compare_checkpoints(args.args[0], args.args[1])
    
    elif args.command == 'export':
        if len(args.args) < 2:
            print("Error: Checkpoint name and output path required")
            print("Usage: python checkpoint_manager.py export <checkpoint_name> <output.pt>")
            return
        
        manager.export_weights(args.args[0], args.args[1])
    
    elif args.command == 'cleanup':
        manager.cleanup_old_checkpoints(
            keep_recent=args.keep,
            dry_run=not args.no_dry_run
        )


if __name__ == "__main__":
    main()
