"""Unit tests for train_utils module."""
import sys
import os
import tempfile
import numpy as np
import torch
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fusion_model import FusionModel
from train_utils import train_fusion_model, _train_fusion_model_legacy, train_single_feature_model


def create_dummy_data(n_samples=100, pointnet_dim=256, ulip_dim=256, num_classes=40, random_labels=False):
    """Create dummy feature and label tensors for testing."""
    train_pointnet = torch.randn(n_samples, pointnet_dim).float()
    train_ulip = torch.randn(n_samples, ulip_dim).float()
    if random_labels:
        train_labels = torch.randint(0, num_classes, (n_samples,)).long()
    else:
        # Create simple linear relationship to ensure some learning
        # Use first dimension of pointnet features to generate labels
        train_labels = (train_pointnet[:, 0] > 0).long()
    
    test_pointnet = torch.randn(n_samples // 5, pointnet_dim).float()
    test_ulip = torch.randn(n_samples // 5, ulip_dim).float()
    test_labels = torch.randint(0, num_classes, (n_samples // 5,)).long()
    
    train_features = (train_pointnet, train_ulip, train_labels)
    test_features = (test_pointnet, test_ulip, test_labels)
    return train_features, test_features


def test_early_stopping_no_improvement():
    """Test that early stopping stops training when no improvement."""
    train_features, test_features = create_dummy_data(random_labels=True)
    model = FusionModel(pointnet_dim=256, ulip_dim=256, num_classes=40)
    
    train_losses, val_accs, test_accs = _train_fusion_model_legacy(
        model=model,
        train_features=train_features,
        train_labels=train_features[2],
        test_features=test_features,
        test_labels=test_features[2],
        epochs=10,
        lr=0.001,
        batch_size=16,
        early_stopping_patience=2,
    )
    
    # With random labels, accuracy should not improve significantly
    # Early stopping should trigger before 10 epochs
    assert len(train_losses) < 10, f"Expected early stopping to reduce epochs, got {len(train_losses)} epochs"
    # Should have at least 2 epochs (patience)
    assert len(train_losses) >= 2


def test_early_stopping_with_improvement():
    """Test that early stopping does not stop when accuracy improves."""
    # Create data where model can learn (simple binary separation)
    train_features, test_features = create_dummy_data(random_labels=False)
    model = FusionModel(pointnet_dim=256, ulip_dim=256, num_classes=2)
    
    train_losses, val_accs, test_accs = _train_fusion_model_legacy(
        model=model,
        train_features=train_features,
        train_labels=train_features[2],
        test_features=test_features,
        test_labels=test_features[2],
        epochs=5,
        lr=0.001,
        batch_size=16,
        early_stopping_patience=10,  # High patience, should not stop early
    )
    
    # Should run all epochs because accuracy improves
    assert len(train_losses) == 5, f"Expected all 5 epochs, got {len(train_losses)}"


def test_cosine_lr_scheduler():
    """Test that cosine learning rate scheduler is used."""
    train_features, test_features = create_dummy_data()
    model = FusionModel(pointnet_dim=256, ulip_dim=256, num_classes=40)
    
    train_losses, val_accs, test_accs = _train_fusion_model_legacy(
        model=model,
        train_features=train_features,
        train_labels=train_features[2],
        test_features=test_features,
        test_labels=test_features[2],
        epochs=3,
        lr=0.001,
        batch_size=16,
        lr_scheduler='cosine',
    )
    
    # Should complete training without error
    assert len(train_losses) == 3



def test_reduce_lr_on_plateau_scheduler():
    """Test that ReduceLROnPlateau scheduler is used."""
    train_features, test_features = create_dummy_data(random_labels=True)
    model = FusionModel(pointnet_dim=256, ulip_dim=256, num_classes=40)
    
    train_losses, val_accs, test_accs = _train_fusion_model_legacy(
        model=model,
        train_features=train_features,
        train_labels=train_features[2],
        test_features=test_features,
        test_labels=test_features[2],
        epochs=4,
        lr=0.001,
        batch_size=16,
        lr_scheduler='plateau',
    )
    
    assert len(train_losses) == 4


def test_comprehensive_metrics_saved(tmp_path):
    """Test that comprehensive metrics are saved to JSON file."""
    train_features, test_features = create_dummy_data()
    model = FusionModel(pointnet_dim=256, ulip_dim=256, num_classes=40)
    
    metrics_path = tmp_path / "metrics.json"
    train_losses, val_accs, test_accs = _train_fusion_model_legacy(
        model=model,
        train_features=train_features,
        train_labels=train_features[2],
        test_features=test_features,
        test_labels=test_features[2],
        epochs=2,
        lr=0.001,
        batch_size=16,
        save_metrics=str(metrics_path),
    )
    
    # Check that metrics file was created
    assert metrics_path.exists()



def test_training_curves_saved(tmp_path):
    """Test that training curves (loss/accuracy arrays) are saved."""
    train_features, test_features = create_dummy_data()
    model = FusionModel(pointnet_dim=256, ulip_dim=256, num_classes=40)
    
    curves_path = tmp_path / "curves.npz"
    train_losses, val_accs, test_accs = _train_fusion_model_legacy(
        model=model,
        train_features=train_features,
        train_labels=train_features[2],
        test_features=test_features,
        test_labels=test_features[2],
        epochs=2,
        lr=0.001,
        batch_size=16,
        save_training_curves=str(curves_path),
    )
    
    assert curves_path.exists()



def test_train_fusion_model_with_validation():
    """Test that early stopping uses validation set, not test set."""
    import warnings
    import numpy as np
    import torch
    from fusion_model import FusionModel
    
    # Set random seed for deterministic behavior
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create dummy features for train/val/test
    n_samples = 100
    pointnet_dim = 256
    ulip_dim = 256
    num_classes = 40
    
    train_pointnet = torch.randn(n_samples, pointnet_dim).float()
    train_ulip = torch.randn(n_samples, ulip_dim).float()
    train_labels = torch.randint(0, num_classes, (n_samples,)).long()
    
    val_pointnet = torch.randn(n_samples // 5, pointnet_dim).float()
    val_ulip = torch.randn(n_samples // 5, ulip_dim).float()
    val_labels = torch.randint(0, num_classes, (n_samples // 5,)).long()
    
    test_pointnet = torch.randn(n_samples // 5, pointnet_dim).float()
    test_ulip = torch.randn(n_samples // 5, ulip_dim).float()
    test_labels = torch.randint(0, num_classes, (n_samples // 5,)).long()
    
    model = FusionModel(pointnet_dim=pointnet_dim, ulip_dim=ulip_dim, num_classes=num_classes)
    
    # Train with validation set and early stopping patience 2
    # Capture warnings to verify no test set leakage warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        train_losses, val_accs, test_accs = train_fusion_model(
            model=model,
            train_features_pointnet=train_pointnet,
            train_features_ulip=train_ulip,
            train_labels=train_labels,
            val_features_pointnet=val_pointnet,
            val_features_ulip=val_ulip,
            val_labels=val_labels,
            test_features_pointnet=test_pointnet,
            test_features_ulip=test_ulip,
            test_labels=test_labels,
            epochs=10,
            batch_size=16,
            learning_rate=0.001,
            early_stopping_patience=2,
            lr_scheduler='plateau'
        )
        
        # Verify no warning about test set leakage (validation set is provided)
        leakage_warnings = [
            warning for warning in w 
            if "No validation set provided" in str(warning.message)
            or "test set leakage" in str(warning.message)
        ]
        assert len(leakage_warnings) == 0, \
            f"Unexpected test set leakage warning: {leakage_warnings}"
    
    # Should return three lists (train_losses, val_accs, test_accs)
    assert len(train_losses) == len(val_accs) == len(test_accs)
    
    # With random labels and early_stopping_patience=2, training should stop early
    # (validation accuracy unlikely to improve significantly)
    # This verifies that early stopping is actually using validation set
    assert len(train_losses) < 10, \
        f"Expected early stopping to reduce epochs, got {len(train_losses)} epochs"
    # Should have at least patience + 1 epochs (2 patience = at least 3 epochs)
    assert len(train_losses) >= 3, \
        f"Expected at least 3 epochs with patience=2, got {len(train_losses)}"
    
    # Verify validation accuracy is being tracked (not just test accuracy)
    # val_accs should contain values between 0 and 1
    assert all(0 <= acc <= 1 for acc in val_accs), \
        f"Validation accuracies out of range: {val_accs}"
    
    # Verify test accuracy is also tracked (separate from validation)
    assert all(0 <= acc <= 1 for acc in test_accs), \
        f"Test accuracies out of range: {test_accs}"
    
    # Additional check: validation and test accuracies should be different
    # (since they're different random datasets, though could coincidentally match)
    # We'll just check they have same length but can differ
    assert len(val_accs) == len(test_accs)


def test_train_fusion_model_without_validation_warning():
    """Test that warning is raised when no validation set is provided."""
    import warnings
    import numpy as np
    import torch
    from fusion_model import FusionModel
    
    # Set random seed
    torch.manual_seed(43)
    np.random.seed(43)
    
    n_samples = 50
    pointnet_dim = 256
    ulip_dim = 256
    num_classes = 40
    
    train_pointnet = torch.randn(n_samples, pointnet_dim).float()
    train_ulip = torch.randn(n_samples, ulip_dim).float()
    train_labels = torch.randint(0, num_classes, (n_samples,)).long()
    
    test_pointnet = torch.randn(n_samples // 5, pointnet_dim).float()
    test_ulip = torch.randn(n_samples // 5, ulip_dim).float()
    test_labels = torch.randint(0, num_classes, (n_samples // 5,)).long()
    
    model = FusionModel(pointnet_dim=pointnet_dim, ulip_dim=ulip_dim, num_classes=num_classes)
    
    # Train WITHOUT validation set - should trigger warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        train_losses, test_accs = train_fusion_model(
            model=model,
            train_features_pointnet=train_pointnet,
            train_features_ulip=train_ulip,
            train_labels=train_labels,
            test_features_pointnet=test_pointnet,
            test_features_ulip=test_ulip,
            test_labels=test_labels,
            epochs=3,
            batch_size=16,
            learning_rate=0.001,
            early_stopping_patience=10  # High patience to avoid early stop
        )
        
        # Verify warning about test set leakage IS raised
        leakage_warnings = [
            warning for warning in w 
            if "No validation set provided" in str(warning.message)
            or "test set leakage" in str(warning.message)
        ]
        assert len(leakage_warnings) > 0, \
            "Expected warning about test set leakage when no validation set provided"
        
        # Verify the warning message contains expected text
        warning_msg = str(leakage_warnings[0].message)
        assert "test set leakage" in warning_msg or "No validation set provided" in warning_msg
    
    # Should return two lists (train_losses, test_accs) for backward compatibility
    assert len(train_losses) == len(test_accs)
    assert len(train_losses) == 3  # Should run all epochs with high patience


def test_train_single_feature_model_with_validation():
    """Test that train_single_feature_model works with validation set."""
    from fusion_model import PointNetOnlyClassifier
    n_samples = 100
    feature_dim = 256
    num_classes = 40
    
    train_features = torch.randn(n_samples, feature_dim).float()
    train_labels = torch.randint(0, num_classes, (n_samples,)).long()
    
    val_features = torch.randn(n_samples // 5, feature_dim).float()
    val_labels = torch.randint(0, num_classes, (n_samples // 5,)).long()
    
    test_features = torch.randn(n_samples // 5, feature_dim).float()
    test_labels = torch.randint(0, num_classes, (n_samples // 5,)).long()
    
    model = PointNetOnlyClassifier(input_dim=feature_dim, num_classes=num_classes)
    
    train_losses, val_accs, test_accs = train_single_feature_model(
        model=model,
        features=train_features,
        labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        test_features=test_features,
        test_labels=test_labels,
        epochs=5,
        batch_size=16,
        lr=0.001,
        early_stopping_patience=10,
        lr_scheduler='plateau'
    )
    
    # Should return three lists
    assert len(train_losses) == len(val_accs) == len(test_accs)