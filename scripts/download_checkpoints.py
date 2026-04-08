#!/usr/bin/env python
# scripts/download_checkpoints.py
"""
Download required checkpoint files for the fusion pipeline.
Supports downloading ULIP-2 checkpoint from Hugging Face Hub.
Optionally downloads OpenCLIP checkpoint.
"""

import os
import sys
import argparse
import hashlib
import shutil
import torch
from pathlib import Path
from typing import Optional, Tuple

# Progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False



def download_hf(repo_id: str, filename: str, dest: Path, force: bool = False) -> bool:
    """Download a file from Hugging Face Hub."""
    if dest.exists() and not force:
        print(f"File already exists: {dest}")
        return True
    
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(f"ERROR: huggingface_hub not installed. Install with: pip install huggingface-hub")
        print(f"Alternatively, download manually from: https://huggingface.co/{repo_id}/blob/main/{filename}")
        return False
    
    print(f"Downloading {filename} from {repo_id}...")
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=dest.parent,
            force_download=force,
            local_files_only=False
        )
        # hf_hub_download returns path in cache; move to desired location
        shutil.move(downloaded, dest)
        print(f"Downloaded to {dest}")
        return True
    except Exception as e:
        print(f"Failed to download from Hugging Face: {e}")
        return False

def download_ulip_checkpoint(output_dir: Path, force: bool = False) -> Optional[Path]:
    """Download ULIP-2 checkpoint."""
    # Two possible filenames; config expects the long name
    hf_filename = "ulip2_pointbert_weights.pth"
    target_filename = "ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / target_filename
    hf_path = output_dir / hf_filename
    
    # If target already exists, return
    if target_path.exists() and not force:
        print(f"ULIP-2 checkpoint already exists: {target_path}")
        return target_path
    
    # Try Hugging Face download
    success = download_hf("salesforce/ULIP-2", hf_filename, hf_path, force)
    if not success:
        # Fallback to direct URL download
        print("Hugging Face download failed. Trying direct URL download...")
        try:
            import requests
            url = f"https://huggingface.co/salesforce/ULIP-2/resolve/main/{hf_filename}"
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            with open(hf_path, 'wb') as f:
                if HAS_TQDM and total_size:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=hf_filename) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    print(f"Downloading {hf_filename}...")
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Downloaded via direct URL: {hf_path}")
            success = True
        except Exception as e:
            print(f"Direct URL download also failed: {e}")
            success = False
    
    # Rename to target filename if needed
    if hf_path.exists() and not target_path.exists():
        shutil.move(hf_path, target_path)
        print(f"Renamed to {target_filename}")
    
    if target_path.exists():
        print(f"ULIP-2 checkpoint ready at {target_path}")
        return target_path
    else:
        print("ERROR: Failed to download ULIP-2 checkpoint.")
        print("Please download manually:")
        print(f"  - Hugging Face: https://huggingface.co/salesforce/ULIP-2/blob/main/{hf_filename}")
        print(f"  - Official repository: https://github.com/salesforce/ULIP")
        print(f"Place the file at {target_path}")
        return None

def download_openclip_checkpoint(output_dir: Path, force: bool = False) -> Optional[Path]:
    """Download OpenCLIP checkpoint (optional)."""
    target_filename = "open_clip_pytorch_model.bin"
    target_path = output_dir / target_filename
    
    if target_path.exists() and not force:
        print(f"OpenCLIP checkpoint already exists: {target_path}")
        return target_path
    
    # OpenCLIP checkpoint can be downloaded from Hugging Face or via open_clip package
    # We'll attempt to use open_clip's pretrained downloader
    try:
        import open_clip
        print("Attempting to download OpenCLIP ViT-g/14 via open_clip...")
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
        # Save model state dict
        torch.save(model.state_dict(), target_path)
        print(f"Saved OpenCLIP checkpoint to {target_path}")
        return target_path
    except Exception as e:
        print(f"Failed to download via open_clip: {e}")
        print("You can manually download OpenCLIP checkpoint:")
        print("  - Use open_clip package: `model, _, _ = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')`")
        print(f"  - Save model.state_dict() to {target_path}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download checkpoint files for fusion pipeline")
    parser.add_argument("--output_dir", "-o", type=str, default="checkpoints",
                        help="Directory to save checkpoints (default: checkpoints)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force download even if file exists")
    parser.add_argument("--skip_ulip", action="store_true",
                        help="Skip ULIP-2 checkpoint download")
    parser.add_argument("--skip_openclip", action="store_true",
                        help="Skip OpenCLIP checkpoint download")
    parser.add_argument("--list", action="store_true",
                        help="List required checkpoint files and exit")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir).absolute()
    
    if args.list:
        print("Required checkpoint files:")
        print("  1. ULIP-2 checkpoint: ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt")
        print("     Source: Hugging Face salesforce/ULIP-2 (ulip2_pointbert_weights.pth)")
        print("  2. OpenCLIP checkpoint: open_clip_pytorch_model.bin (optional)")
        print("     Source: open_clip pretrained model 'ViT-g-14'")
        return 0
    
    print(f"Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success = True
    if not args.skip_ulip:
        print("\n=== Downloading ULIP-2 checkpoint ===")
        ulip_path = download_ulip_checkpoint(output_dir, args.force)
        if ulip_path is None:
            success = False
    
    if not args.skip_openclip:
        print("\n=== Downloading OpenCLIP checkpoint (optional) ===")
        clip_path = download_openclip_checkpoint(output_dir, args.force)
        if clip_path is None:
            print("OpenCLIP checkpoint not downloaded; pipeline will download weights on first run.")
    
    if success:
        print("\n✓ Checkpoint download completed.")
        print(f"  Checkpoints saved in: {output_dir}")
        print("\nNext steps:")
        print("  1. Verify config/fusion_config.yaml points to the correct checkpoint paths")
        print("  2. Run feature extraction: python scripts/extract_features.py")
        return 0
    else:
        print("\n✗ Some downloads failed. See instructions above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())