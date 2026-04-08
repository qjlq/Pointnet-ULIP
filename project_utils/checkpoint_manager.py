import os
import torch
import json
from typing import Dict, Any

class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, epoch: int, model: torch.nn.Module, 
                        optimizer: torch.optim.Optimizer, loss: float, 
                        accuracy: float, is_best: bool = False, 
                        extra_info: Dict[str, Any] = None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
            'extra_info': extra_info or {}
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_name: str) -> Dict[str, Any]:
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        return torch.load(checkpoint_path, map_location='cpu')
    
    def get_latest_checkpoint(self):
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_epoch_')]
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return checkpoints[-1]