# Updated: 2026-03-27 (fusion pipeline integration)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import warnings
from typing import Tuple, Optional, List
from project_utils.checkpoint_manager import CheckpointManager

def train_fusion_model(model: nn.Module, *args, **kwargs) -> Tuple[List[float], List[float]]:
    """Train fusion model on extracted features with backward compatibility.
    
    Supports two calling signatures:
    
    1. Old signature (backward compatible):
        train_fusion_model(model, train_features, train_labels, test_features, test_labels,
                          epochs=15, lr=0.001, batch_size=16)
        where train_features and test_features are tuples of (geo_features, vlm_features, labels).
        The separate train_labels and test_labels are deprecated and ignored.
    
    2. New signature (pipeline compatibility):
        train_fusion_model(model,
                          train_features_pointnet, train_features_ulip, train_labels,
                          test_features_pointnet, test_features_ulip, test_labels,
                          epochs=50, batch_size=32, learning_rate=0.001,
                          checkpoint_dir='checkpoints', save_interval=10, **kwargs)
        where features can be numpy arrays or torch tensors.
    
    Args:
        model: FeatureFusionHead or FusionModel to train
        *args: Positional arguments for old signature
        **kwargs: Keyword arguments for both signatures
        
    Returns:
        Tuple of (train_losses, test_accuracies) per epoch
    """
    # --- Signature detection ---
    if 'train_features_pointnet' in kwargs:
        # New signature: all arguments passed as kwargs
        train_features_pointnet = kwargs.pop('train_features_pointnet')
        train_features_ulip = kwargs.pop('train_features_ulip')
        train_labels = kwargs.pop('train_labels')
        test_features_pointnet = kwargs.pop('test_features_pointnet')
        test_features_ulip = kwargs.pop('test_features_ulip')
        test_labels = kwargs.pop('test_labels')
        
        # Convert numpy arrays to torch tensors if needed
        def _to_tensor(x):
            if isinstance(x, np.ndarray):
                return torch.FloatTensor(x)
            return x
        
        train_features_pointnet = _to_tensor(train_features_pointnet)
        train_features_ulip = _to_tensor(train_features_ulip)
        train_labels = _to_tensor(train_labels).long()
        test_features_pointnet = _to_tensor(test_features_pointnet)
        test_features_ulip = _to_tensor(test_features_ulip)
        test_labels = _to_tensor(test_labels).long()
        
        # Create tuples in old format
        train_features = (train_features_pointnet, train_features_ulip, train_labels)
        test_features = (test_features_pointnet, test_features_ulip, test_labels)
        
        # Map learning_rate to lr for backward compatibility
        if 'learning_rate' in kwargs:
            kwargs['lr'] = kwargs.pop('learning_rate')
        
        # Call legacy function with tuple format
        return _train_fusion_model_legacy(model, train_features, train_labels,
                                         test_features, test_labels, **kwargs)
    
    else:
        # Old signature: either positional args or keyword args
        train_features = None
        train_labels = None
        test_features = None
        test_labels = None
        
        # Check if passed as keyword arguments
        if 'train_features' in kwargs:
            train_features = kwargs.pop('train_features')
            train_labels = kwargs.pop('train_labels')
            test_features = kwargs.pop('test_features')
            test_labels = kwargs.pop('test_labels')
        elif len(args) >= 4:
            # Positional arguments
            train_features, train_labels, test_features, test_labels = args[:4]
        else:
            raise ValueError(
                "Invalid arguments. Either use new signature with keyword arguments "
                "(train_features_pointnet, etc.) or old signature with 4 positional arguments "
                "or keyword arguments (train_features, train_labels, test_features, test_labels)."
            )
        
        # Pass any remaining kwargs (epochs, lr, batch_size)
        return _train_fusion_model_legacy(model, train_features, train_labels,
                                         test_features, test_labels, **kwargs)


def _train_fusion_model_legacy(model: nn.Module,
                               train_features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                               train_labels: torch.Tensor,
                               test_features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                               test_labels: torch.Tensor,
                               epochs: int = 15,
                               lr: float = 0.001,
                               batch_size: int = 16,
                               checkpoint_dir: Optional[str] = None,
                               save_interval: int = 10,
                               **kwargs) -> Tuple[List[float], List[float]]:
    """Legacy training function (internal)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Unpack features
    f_geo_train, f_vlm_train, labels_train = train_features
    f_geo_test, f_vlm_test, labels_test = test_features
    
    # Verify shapes
    assert f_geo_train.shape[0] == f_vlm_train.shape[0] == labels_train.shape[0], \
        "Training feature batch sizes must match"
    assert f_geo_test.shape[0] == f_vlm_test.shape[0] == labels_test.shape[0], \
        "Test feature batch sizes must match"
    
    # Create TensorDataset and DataLoader
    train_dataset = TensorDataset(f_geo_train, f_vlm_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Checkpoint management
    checkpoint_manager = None
    if checkpoint_dir is not None:
        checkpoint_manager = CheckpointManager(checkpoint_dir)
    best_accuracy = 0.0

    train_losses, test_accs = [], []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for f_geo_batch, f_vlm_batch, labels_batch in pbar:
            f_geo_batch = f_geo_batch.to(device)
            f_vlm_batch = f_vlm_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(f_geo_batch, f_vlm_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            f_geo_test_dev = f_geo_test.to(device)
            f_vlm_test_dev = f_vlm_test.to(device)
            labels_test_dev = labels_test.to(device)
            
            outputs = model(f_geo_test_dev, f_vlm_test_dev)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels_test_dev).sum().item()
            accuracy = correct / len(labels_test_dev)
            test_accs.append(accuracy)
        
        # Checkpoint saving
        if checkpoint_manager is not None:
            is_best = accuracy > best_accuracy
            if is_best:
                best_accuracy = accuracy
            
            # Save checkpoint at save_interval intervals or final epoch
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
                checkpoint_manager.save_checkpoint(
                    epoch=epoch + 1,
                    model=model,
                    optimizer=optimizer,
                    loss=avg_loss,
                    accuracy=accuracy,
                    is_best=is_best,
                    extra_info={'train_loss': avg_loss, 'test_accuracy': accuracy}
                )
        
        print(f'Epoch {epoch+1:3d}: Loss = {avg_loss:.4f}, Test Acc = {accuracy:.4f}')
    
    return train_losses, test_accs
def create_feature_tensors(geo_features: torch.Tensor, 
                          vlm_features: torch.Tensor, 
                          labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create tuple of feature tensors for training.
    
    Args:
        geo_features: Geometric features tensor (N, geo_dim)
        vlm_features: Semantic features tensor (N, vlm_dim)
        labels: Labels tensor (N,)
        
    Returns:
        Tuple of (geo_features, vlm_features, labels)
    """
    # Ensure same device and dtype
    geo_features = geo_features.float()
    vlm_features = vlm_features.float()
    labels = labels.long()
    
    return geo_features, vlm_features, labels

def split_features(features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                   train_idx: torch.Tensor,
                   test_idx: torch.Tensor) -> Tuple:
    """Split features into train and test sets.
    
    Args:
        features: Tuple of (geo_features, vlm_features, labels)
        train_idx: Indices for training set
        test_idx: Indices for test set
    
    Returns:
        Tuple of (train_features, test_features) where each is a tuple of
        (geo_features, vlm_features, labels)
    """
    f_geo, f_vlm, labels = features
    
    f_geo_train = f_geo[train_idx]
    f_vlm_train = f_vlm[train_idx]
    labels_train = labels[train_idx]
    
    f_geo_test = f_geo[test_idx]
    f_vlm_test = f_vlm[test_idx]
    labels_test = labels[test_idx]
    
    train_features = (f_geo_train, f_vlm_train, labels_train)
    test_features = (f_geo_test, f_vlm_test, labels_test)
    
    return train_features, test_features


def train_single_feature_model(model: nn.Module,
                               features: torch.Tensor,
                               labels: torch.Tensor,
                               test_features: torch.Tensor,
                               test_labels: torch.Tensor,
                               epochs: int = 50,
                               lr: float = 0.001,
                               batch_size: int = 32,
                               checkpoint_dir: Optional[str] = None,
                               save_interval: int = 10,
                               **kwargs) -> Tuple[List[float], List[float]]:
    """Train a model using single feature type.
    
    Args:
        model: Classification model (e.g., PointNetOnlyClassifier, ULIPOnlyClassifier)
        features: Training features tensor of shape (N, feature_dim)
        labels: Training labels tensor of shape (N,)
        test_features: Test features tensor of shape (M, feature_dim)
        test_labels: Test labels tensor of shape (M,)
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
        checkpoint_dir: Directory for saving checkpoints
        save_interval: Save checkpoint every N epochs
        **kwargs: Additional keyword arguments
    
    Returns:
        Tuple of (train_losses, test_accuracies) per epoch
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Verify shapes
    assert features.shape[0] == labels.shape[0], "Training feature and label batch sizes must match"
    assert test_features.shape[0] == test_labels.shape[0], "Test feature and label batch sizes must match"
    
    # Create TensorDataset and DataLoader
    train_dataset = TensorDataset(features, labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Checkpoint management
    checkpoint_manager = None
    if checkpoint_dir is not None:
        checkpoint_manager = CheckpointManager(checkpoint_dir)
    best_accuracy = 0.0
    
    train_losses, test_accs = [], []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for features_batch, labels_batch in pbar:
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(features_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_features_dev = test_features.to(device)
            test_labels_dev = test_labels.to(device)
            
            outputs = model(test_features_dev)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == test_labels_dev).sum().item()
            accuracy = correct / len(test_labels_dev)
            test_accs.append(accuracy)
        
        # Checkpoint saving
        if checkpoint_manager is not None:
            is_best = accuracy > best_accuracy
            if is_best:
                best_accuracy = accuracy
            
            # Save checkpoint at save_interval intervals or final epoch
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
                checkpoint_manager.save_checkpoint(
                    epoch=epoch + 1,
                    model=model,
                    optimizer=optimizer,
                    loss=avg_loss,
                    accuracy=accuracy,
                    is_best=is_best,
                    extra_info={'train_loss': avg_loss, 'test_accuracy': accuracy}
                )
        
        print(f'Epoch {epoch+1:3d}: Loss = {avg_loss:.4f}, Test Acc = {accuracy:.4f}')
    
    return train_losses, test_accs