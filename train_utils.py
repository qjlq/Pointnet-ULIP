# Updated: 2026-03-27 (fusion pipeline integration)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import warnings
from typing import Tuple, Optional, List, Callable, Union
from project_utils.checkpoint_manager import CheckpointManager
try:
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    confusion_matrix = f1_score = accuracy_score = None

def train_fusion_model(model: nn.Module, *args, **kwargs) -> Union[Tuple[List[float], List[float]], Tuple[List[float], List[float], List[float]]]:
    """Train fusion model on extracted features with backward compatibility.
    
    Supports two calling signatures:
    
    1. Old signature (backward compatible):
        train_fusion_model(model, train_features, train_labels, test_features, test_labels,
                          epochs=15, lr=0.001, batch_size=16)
        where train_features and test_features are tuples of (geo_features, vlm_features, labels).
        The separate train_labels and test_labels are deprecated and ignored.
        Returns (train_losses, test_accuracies) per epoch.
    
    2. New signature (pipeline compatibility):
        train_fusion_model(model,
                          train_features_pointnet, train_features_ulip, train_labels,
                          val_features_pointnet, val_features_ulip, val_labels,
                          test_features_pointnet, test_features_ulip, test_labels,
                          epochs=50, batch_size=32, learning_rate=0.001,
                          checkpoint_dir='checkpoints', save_interval=10, **kwargs)
        where features can be numpy arrays or torch tensors.
        If validation set (val_features_*) is provided, returns (train_losses, val_accuracies, test_accuracies).
        If validation set is not provided, returns (train_losses, test_accuracies) for backward compatibility.
    
    Additional training options (passed via **kwargs):
        lr_scheduler: Optional[str] - 'cosine' or 'plateau' learning rate scheduler
        early_stopping_patience: Optional[int] - patience for early stopping
        save_metrics: Optional[str] - path to save comprehensive metrics JSON
        save_training_curves: Optional[str] - path to save training curves NPZ
        epoch_callback: Optional[Callable] - callback function called after each epoch with signature (epoch_idx, train_loss, test_accuracy, learning_rate)
    
    Args:
        model: FeatureFusionHead or FusionModel to train
        *args: Positional arguments for old signature
        **kwargs: Keyword arguments for both signatures
        
    Returns:
        Tuple of (train_losses, val_accuracies, test_accuracies) per epoch if validation set provided,
        otherwise tuple of (train_losses, test_accuracies).
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
        
        # Extract validation features if provided, otherwise default to None
        val_features_pointnet = kwargs.pop('val_features_pointnet', None)
        val_features_ulip = kwargs.pop('val_features_ulip', None)
        val_labels = kwargs.pop('val_labels', None)
        
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
        
        # Convert validation tensors if provided
        if val_features_pointnet is not None:
            val_features_pointnet = _to_tensor(val_features_pointnet)
            val_features_ulip = _to_tensor(val_features_ulip)
            val_labels = _to_tensor(val_labels).long()
        
        # Create tuples in old format
        train_features = (train_features_pointnet, train_features_ulip, train_labels)
        test_features = (test_features_pointnet, test_features_ulip, test_labels)
        val_features = None
        if val_features_pointnet is not None:
            val_features = (val_features_pointnet, val_features_ulip, val_labels)
        
        # Map learning_rate to lr for backward compatibility
        if 'learning_rate' in kwargs:
            kwargs['lr'] = kwargs.pop('learning_rate')
        
        # Call legacy function with tuple format
        train_losses, val_accs, test_accs = _train_fusion_model_legacy(
            model, train_features, train_labels,
            test_features, test_labels,
            val_features=val_features,
            val_labels=val_labels,
            **kwargs
        )
        
        # Return appropriate number of lists based on whether validation set was provided
        if val_features is None:
            # No validation set: return train_losses and test_accs for backward compatibility
            return train_losses, test_accs
        else:
            # Validation set provided: return all three lists
            return train_losses, val_accs, test_accs
    
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
        train_losses, val_accs, test_accs = _train_fusion_model_legacy(
            model, train_features, train_labels,
            test_features, test_labels, **kwargs
        )
        # Old signature always returns two lists for backward compatibility
        return train_losses, test_accs


def _train_fusion_model_legacy(model: nn.Module,
                               train_features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                               train_labels: torch.Tensor,
                               test_features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                               test_labels: torch.Tensor,
                               val_features: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                               val_labels: Optional[torch.Tensor] = None,
                               epochs: int = 15,
                               lr: float = 0.001,
                               batch_size: int = 16,
                               checkpoint_dir: Optional[str] = None,
                               save_interval: int = 10,
                               **kwargs) -> Tuple[List[float], List[float], List[float]]:
    """Legacy training function (internal)."""
    # Extract new training parameters from kwargs
    lr_scheduler = kwargs.pop('lr_scheduler', None)
    lr_scheduler_kwargs = kwargs.pop('lr_scheduler_kwargs', {})
    early_stopping_patience = kwargs.pop('early_stopping_patience', None)
    early_stopping_min_delta = kwargs.pop('early_stopping_min_delta', 0.0)
    save_metrics = kwargs.pop('save_metrics', None)
    save_training_curves = kwargs.pop('save_training_curves', None)
    epoch_callback = kwargs.pop('epoch_callback', None)
    # Warn about unrecognized kwargs
    if kwargs:
        warnings.warn(f"Unrecognized keyword arguments: {list(kwargs.keys())}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Unpack features
    f_geo_train, f_vlm_train, labels_train = train_features
    f_geo_test, f_vlm_test, labels_test = test_features
    
    # Determine validation set
    has_validation = val_features is not None
    if not has_validation:
        val_features = test_features
        val_labels = test_labels
        warnings.warn("No validation set provided. Using test set for validation, which may cause test set leakage.")
    
    # Unpack validation features
    f_geo_val, f_vlm_val, labels_val = val_features
    
    # Verify shapes
    if not (f_geo_train.shape[0] == f_vlm_train.shape[0] == labels_train.shape[0]):
        raise ValueError("Training feature batch sizes must match")
    if not (f_geo_test.shape[0] == f_vlm_test.shape[0] == labels_test.shape[0]):
        raise ValueError("Test feature batch sizes must match")
    if not (f_geo_val.shape[0] == f_vlm_val.shape[0] == labels_val.shape[0]):
        raise ValueError("Validation feature batch sizes must match")
    
    # Create TensorDataset and DataLoader
    train_dataset = TensorDataset(f_geo_train, f_vlm_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataset = TensorDataset(f_geo_val, f_vlm_val, labels_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = None
    if lr_scheduler == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, **lr_scheduler_kwargs)
    elif lr_scheduler == 'plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, **lr_scheduler_kwargs)
    elif lr_scheduler is not None:
        warnings.warn(f"Unrecognized lr_scheduler '{lr_scheduler}'. No scheduler will be used.")

    # Checkpoint management
    checkpoint_manager = None
    if checkpoint_dir is not None:
        checkpoint_manager = CheckpointManager(checkpoint_dir)
    best_val_accuracy = 0.0
    best_model_state = None
    early_stopping_counter = 0
    
    train_losses, val_accs, test_accs = [], [], []
    
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
        
        # Evaluate on validation set (for early stopping, LR scheduling, best model)
        model.eval()
        with torch.no_grad():
            f_geo_val_dev = f_geo_val.to(device)
            f_vlm_val_dev = f_vlm_val.to(device)
            labels_val_dev = labels_val.to(device)
            
            outputs = model(f_geo_val_dev, f_vlm_val_dev)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels_val_dev).sum().item()
            val_accuracy = correct / len(labels_val_dev)
            val_accs.append(val_accuracy)
        
        # Evaluate on test set (for reporting only)
        with torch.no_grad():
            f_geo_test_dev = f_geo_test.to(device)
            f_vlm_test_dev = f_vlm_test.to(device)
            labels_test_dev = labels_test.to(device)
            
            outputs = model(f_geo_test_dev, f_vlm_test_dev)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels_test_dev).sum().item()
            test_accuracy = correct / len(labels_test_dev)
            test_accs.append(test_accuracy)
        
        # Call epoch callback if provided (pass test accuracy for backward compatibility)
        if epoch_callback is not None:
            lr = optimizer.param_groups[0]['lr']
            epoch_callback(epoch, avg_loss, test_accuracy, lr)
        
        # Early stopping and best model tracking based on validation accuracy
        is_best = val_accuracy > best_val_accuracy + early_stopping_min_delta
        if is_best:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # Learning rate scheduling based on validation accuracy
        if scheduler is not None:
            if lr_scheduler == 'plateau':
                scheduler.step(val_accuracy)
            else:
                scheduler.step()
        
        # Checkpoint saving
        if checkpoint_manager is not None:
            # Save checkpoint at save_interval intervals or final epoch
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
                checkpoint_manager.save_checkpoint(
                    epoch=epoch + 1,
                    model=model,
                    optimizer=optimizer,
                    loss=avg_loss,
                    accuracy=val_accuracy,
                    is_best=is_best,
                    extra_info={'train_loss': avg_loss, 'val_accuracy': val_accuracy, 'test_accuracy': test_accuracy}
                )
        
        # Early stopping check
        if early_stopping_patience is not None and early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
        
        print(f'Epoch {epoch+1:3d}: Loss = {avg_loss:.4f}, Val Acc = {val_accuracy:.4f}, Test Acc = {test_accuracy:.4f}')
    
    # Restore best model weights if early stopping was used
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Restored best model from epoch with validation accuracy {best_val_accuracy:.4f}')
    
    # Save training curves if requested
    if save_training_curves is not None:
        curves_data = {
            'train_losses': train_losses,
            'val_accuracies': val_accs,
            'test_accuracies': test_accs
        }
        np.savez(save_training_curves, **curves_data)
        print(f'Saved training curves to {save_training_curves}')
    
    # Compute comprehensive metrics if requested (using test set for final evaluation)
    if save_metrics is not None:
        import json
        if not SKLEARN_AVAILABLE:
            warnings.warn("sklearn not available, skipping comprehensive metrics")
            return train_losses, val_accs, test_accs
        model.eval()
        with torch.no_grad():
            f_geo_test_dev = f_geo_test.to(device)
            f_vlm_test_dev = f_vlm_test.to(device)
            labels_test_dev = labels_test.to(device)
            outputs = model(f_geo_test_dev, f_vlm_test_dev)
            _, predicted = torch.max(outputs, 1)
            predicted_np = predicted.cpu().numpy()
            labels_np = labels_test_dev.cpu().numpy()
        
        # Compute metrics
        acc_per_class = []
        num_classes = outputs.shape[1]
        for cls in range(num_classes):
            mask = labels_np == cls
            if mask.any():
                acc = (predicted_np[mask] == cls).mean()
            else:
                acc = 0.0
            acc_per_class.append(float(acc))
        
        conf_matrix = confusion_matrix(labels_np, predicted_np).tolist()
        f1_micro = f1_score(labels_np, predicted_np, average='micro')
        f1_macro = f1_score(labels_np, predicted_np, average='macro')
        f1_weighted = f1_score(labels_np, predicted_np, average='weighted')
        
        metrics = {
            'final_accuracy': float(best_val_accuracy),
            'per_class_accuracy': acc_per_class,
            'confusion_matrix': conf_matrix,
            'f1_scores': {
                'micro': float(f1_micro),
                'macro': float(f1_macro),
                'weighted': float(f1_weighted)
            },
            'train_losses': train_losses,
            'val_accuracies': val_accs,
            'test_accuracies': test_accs,
            'epochs_trained': len(train_losses)
        }
        
        with open(save_metrics, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f'Saved comprehensive metrics to {save_metrics}')
    
    return train_losses, val_accs, test_accs
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
                               val_features: Optional[torch.Tensor] = None,
                               val_labels: Optional[torch.Tensor] = None,
                               epochs: int = 50,
                               lr: float = 0.001,
                               batch_size: int = 32,
                               checkpoint_dir: Optional[str] = None,
                               save_interval: int = 10,
                               **kwargs) -> Union[Tuple[List[float], List[float]], Tuple[List[float], List[float], List[float]]]:
    """Train a model using single feature type.
    
    Args:
        model: Classification model (e.g., PointNetOnlyClassifier, ULIPOnlyClassifier)
        features: Training features tensor of shape (N, feature_dim)
        labels: Training labels tensor of shape (N,)
        test_features: Test features tensor of shape (M, feature_dim)
        test_labels: Test labels tensor of shape (M,)
        val_features: Validation features tensor of shape (K, feature_dim) (optional)
        val_labels: Validation labels tensor of shape (K,) (optional)
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
        checkpoint_dir: Directory for saving checkpoints
        save_interval: Save checkpoint every N epochs
        **kwargs: Additional keyword arguments
    
    Returns:
        Tuple of (train_losses, val_accuracies, test_accuracies) per epoch if validation set provided,
        otherwise tuple of (train_losses, test_accuracies).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Extract new training parameters from kwargs
    lr_scheduler = kwargs.pop('lr_scheduler', None)
    lr_scheduler_kwargs = kwargs.pop('lr_scheduler_kwargs', {})
    early_stopping_patience = kwargs.pop('early_stopping_patience', None)
    early_stopping_min_delta = kwargs.pop('early_stopping_min_delta', 0.0)
    save_metrics = kwargs.pop('save_metrics', None)
    save_training_curves = kwargs.pop('save_training_curves', None)
    # Warn about unrecognized kwargs
    if kwargs:
        warnings.warn(f"Unrecognized keyword arguments: {list(kwargs.keys())}")
    
    # Determine validation set
    has_validation = val_features is not None
    if not has_validation:
        val_features = test_features
        val_labels = test_labels
        warnings.warn("No validation set provided. Using test set for validation, which may cause test set leakage.")
    
    # Verify shapes
    if not (features.shape[0] == labels.shape[0]):
        raise ValueError("Training feature and label batch sizes must match")
    if not (test_features.shape[0] == test_labels.shape[0]):
        raise ValueError("Test feature and label batch sizes must match")
    if not (val_features.shape[0] == val_labels.shape[0]):
        raise ValueError("Validation feature and label batch sizes must match")
    
    # Create TensorDataset and DataLoader
    train_dataset = TensorDataset(features, labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataset = TensorDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = None
    if lr_scheduler == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, **lr_scheduler_kwargs)
    elif lr_scheduler == 'plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, **lr_scheduler_kwargs)
    elif lr_scheduler is not None:
        warnings.warn(f"Unrecognized lr_scheduler '{lr_scheduler}'. No scheduler will be used.")
    # Checkpoint management
    checkpoint_manager = None
    if checkpoint_dir is not None:
        checkpoint_manager = CheckpointManager(checkpoint_dir)
    best_val_accuracy = 0.0
    
    train_losses, val_accs, test_accs = [], [], []
    early_stopping_counter = 0
    
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
        
        # Evaluate on validation set (for early stopping, LR scheduling, best model)
        model.eval()
        with torch.no_grad():
            val_features_dev = val_features.to(device)
            val_labels_dev = val_labels.to(device)
            
            outputs = model(val_features_dev)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == val_labels_dev).sum().item()
            val_accuracy = correct / len(val_labels_dev)
            val_accs.append(val_accuracy)
        
        # Evaluate on test set (for reporting only)
        with torch.no_grad():
            test_features_dev = test_features.to(device)
            test_labels_dev = test_labels.to(device)
            
            outputs = model(test_features_dev)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == test_labels_dev).sum().item()
            test_accuracy = correct / len(test_labels_dev)
            test_accs.append(test_accuracy)
        
        # Early stopping and best model tracking based on validation accuracy
        is_best = val_accuracy > best_val_accuracy + early_stopping_min_delta
        if is_best:
            best_val_accuracy = val_accuracy
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # Learning rate scheduling based on validation accuracy
        if scheduler is not None:
            if lr_scheduler == 'plateau':
                scheduler.step(val_accuracy)
            else:
                scheduler.step()
        
        # Checkpoint saving
        if checkpoint_manager is not None:
            # Save checkpoint at save_interval intervals or final epoch
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
                checkpoint_manager.save_checkpoint(
                    epoch=epoch + 1,
                    model=model,
                    optimizer=optimizer,
                    loss=avg_loss,
                    accuracy=val_accuracy,
                    is_best=is_best,
                    extra_info={'train_loss': avg_loss, 'val_accuracy': val_accuracy, 'test_accuracy': test_accuracy}
                )
        
        # Early stopping check
        if early_stopping_patience is not None and early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
        
        print(f'Epoch {epoch+1:3d}: Loss = {avg_loss:.4f}, Val Acc = {val_accuracy:.4f}, Test Acc = {test_accuracy:.4f}')
    
    if has_validation:
        return train_losses, val_accs, test_accs
    else:
        return train_losses, test_accs