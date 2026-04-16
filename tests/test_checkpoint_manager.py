import sys
sys.path.insert(0, '.')
import torch
import tempfile
from project_utils.checkpoint_manager import CheckpointManager

def test_checkpoint_manager_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        manager.save_checkpoint(
            epoch=1,
            model=model,
            optimizer=optimizer,
            loss=0.5,
            accuracy=0.8,
            is_best=True
        )
        
        checkpoint = manager.load_checkpoint("checkpoint_epoch_1.pth")
        assert checkpoint['epoch'] == 1
        assert checkpoint['loss'] == 0.5
        assert checkpoint['accuracy'] == 0.8
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        
        # Check that best model was saved
        best_checkpoint = manager.load_checkpoint("best_model.pth")
        assert best_checkpoint['epoch'] == 1

def test_checkpoint_manager_get_latest():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        manager.save_checkpoint(
            epoch=1,
            model=model,
            optimizer=optimizer,
            loss=0.5,
            accuracy=0.8,
            is_best=False
        )
        
        manager.save_checkpoint(
            epoch=2,
            model=model,
            optimizer=optimizer,
            loss=0.4,
            accuracy=0.85,
            is_best=False
        )
        
        latest = manager.get_latest_checkpoint()
        assert latest == "checkpoint_epoch_2.pth"
        
        checkpoint = manager.load_checkpoint(latest)
        assert checkpoint['epoch'] == 2
        assert checkpoint['loss'] == 0.4

def test_checkpoint_manager_extra_info():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        extra_info = {"val_loss": 0.3, "val_acc": 0.9, "timestamp": "2026-03-28"}
        manager.save_checkpoint(
            epoch=3,
            model=model,
            optimizer=optimizer,
            loss=0.6,
            accuracy=0.75,
            is_best=False,
            extra_info=extra_info
        )
        
        checkpoint = manager.load_checkpoint("checkpoint_epoch_3.pth")
        assert checkpoint['extra_info']['val_loss'] == 0.3
        assert checkpoint['extra_info']['timestamp'] == "2026-03-28"