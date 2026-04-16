import sys
import os
import logging
import threading
import torch
import torch.nn as nn
from typing import Optional, Tuple


def _create_openclip_monkey_patch(openclip_checkpoint: str) -> tuple:
    """Create temporary monkey-patch for open_clip.create_model_and_transforms.
    
    Only patches calls with model_name='ViT-bigG-14' and pretrained='laion2b_s39b_b160k'.
    If pretrained is already a valid file path, does not replace it.
    """
    import open_clip
    
    original_create = open_clip.create_model_and_transforms
    
    def patched_create(model_name, pretrained=None, **kwargs):
        # Only patch specific model/pretrained combination
        if (model_name == 'ViT-bigG-14' and 
            pretrained == 'laion2b_s39b_b160k'):
            # If pretrained is already a valid file path, keep it (should not happen)
            if not (isinstance(pretrained, str) and os.path.isfile(pretrained)):
                pretrained = openclip_checkpoint
        return original_create(model_name, pretrained, **kwargs)
    
    return original_create, patched_create


def _create_pointnet2ops_monkey_patch() -> tuple:
    """Create monkey-patch for pointnet2_ops module to avoid CUDA compilation.
    
    Replaces pointnet2_ops.pointnet2_utils with pure PyTorch implementations
    from ULIP's internal pointmlp module for furthest_point_sample and
    a torch.gather-based gather_operation.
    
    Returns:
        Tuple of (original_module_or_none, restore_function)
    """
    import sys
    import torch
    
    # Import the pure PyTorch furthest_point_sample from ULIP's pointmlp
    furthest_point_sample = None
    try:
        # Try to import from ULIP's pointmlp module
        from models.pointmlp.pointMLP import furthest_point_sample
    except ImportError:
        # Fallback: implement our own pure PyTorch version
        def furthest_point_sample_fallback(xyz, npoint):
            device = xyz.device
            B, N, C = xyz.shape
            centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
            distance = torch.ones(B, N).to(device) * 1e10
            farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
            batch_indices = torch.arange(B, dtype=torch.long).to(device)
            for i in range(npoint):
                centroids[:, i] = farthest
                centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
                dist = torch.sum((xyz - centroid) ** 2, -1)
                distance = torch.min(distance, dist)
                farthest = torch.max(distance, -1)[1]
            return centroids
        furthest_point_sample = furthest_point_sample_fallback
    
    assert furthest_point_sample is not None, 'furthest_point_sample not defined'
    # Implement gather_operation using torch.gather
    def gather_operation(features, idx):
        """
        Args:
            features: (B, C, N) tensor
            idx: (B, npoint) tensor of indices
        Returns:
            (B, C, npoint) tensor
        """
        B, C, N = features.shape
        _, npoint = idx.shape
        # Expand idx to match features dimensions
        idx_expanded = idx.unsqueeze(1).expand(-1, C, -1)  # (B, C, npoint)
        return torch.gather(features, 2, idx_expanded)
    
    # Capture functions to avoid scoping issues
    fps_fn = furthest_point_sample
    gather_op = gather_operation
    
    # Create mock pointnet2_utils module
    class MockPointnet2Utils:
        furthest_point_sample = staticmethod(fps_fn)
        gather_operation = staticmethod(gather_op)
        # Add other functions that might be needed (stubs)
        def __getattr__(self, name):
            # Return a dummy function for any other attribute access
            logging.warning(f"pointnet2_utils.{name} called but not implemented, returning dummy")
            return lambda *args, **kwargs: None
    
    # Create mock pointnet2_ops module
    class MockPointnet2Ops:
        pointnet2_utils = MockPointnet2Utils()
    
    # Store original modules if they exist
    original_pointnet2_ops = sys.modules.get('pointnet2_ops')
    original_pointnet2_utils = sys.modules.get('pointnet2_ops.pointnet2_utils')
    
    # Install our mock modules
    sys.modules['pointnet2_ops'] = MockPointnet2Ops()
    sys.modules['pointnet2_ops.pointnet2_utils'] = MockPointnet2Utils()
    
    def restore():
        """Restore original modules."""
        if original_pointnet2_ops is not None:
            sys.modules['pointnet2_ops'] = original_pointnet2_ops
        else:
            sys.modules.pop('pointnet2_ops', None)
        
        if original_pointnet2_utils is not None:
            sys.modules['pointnet2_ops.pointnet2_utils'] = original_pointnet2_utils
        else:
            sys.modules.pop('pointnet2_ops.pointnet2_utils', None)
    
    return restore


def _create_knn_cuda_monkey_patch():
    """Create monkey-patch for knn_cuda module to avoid CUDA compilation.
    
    Replaces knn_cuda.KNN with a pure PyTorch implementation using
    square distance and topk.
    """
    import sys
    import torch
    
    # Implement square_distance from dvae.py
    def square_distance(src, dst):
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist
    
    # Implement knn_point from dvae.py
    def knn_point(nsample, xyz, new_xyz):
        sqrdists = square_distance(new_xyz, xyz)
        _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
        return group_idx
    
    # Mock KNN class
    class MockKNN:
        def __init__(self, k=1, transpose_mode=False):
            self.k = k
            self.transpose_mode = transpose_mode
        
        def __call__(self, x, y):
            # x: source points [B, N, C]? Actually need to check dvae usage.
            # In dvae they call knn(coor_k, coor_q) where coor_k is source, coor_q is query.
            # The knn_point expects (nsample, xyz, new_xyz) where xyz is all points, new_xyz is query.
            # So we match the same order.
            idx = knn_point(self.k, x, y)
            # Return dummy distances (zeros) and indices
            dist = torch.zeros_like(idx, dtype=torch.float32)
            return dist, idx
    
    # Store original module if exists
    original_knn_cuda = sys.modules.get('knn_cuda')
    
    # Create mock module
    class MockKnnCuda:
        KNN = MockKNN
    
    # Install mock module
    sys.modules['knn_cuda'] = MockKnnCuda()
    
    def restore():
        if original_knn_cuda is not None:
            sys.modules['knn_cuda'] = original_knn_cuda
        else:
            sys.modules.pop('knn_cuda', None)
    
    return restore


def check_gpu_memory_available(device: str = "cuda") -> tuple:
    """Check available GPU memory in bytes and percentage.
    
    Args:
        device: Target device ("cuda" for GPU)
    
    Returns:
        Tuple of (total_memory_bytes, allocated_bytes, free_bytes, free_percentage)
        Returns None for all values if CUDA not available or device is not CUDA.
    """
    if device != "cuda" or not torch.cuda.is_available():
        return None, None, None, None
    
    total = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated(0)
    free = total - allocated
    free_percentage = (free / total) * 100 if total > 0 else 0
    
    return total, allocated, free, free_percentage


def calculate_safe_batch_size(
    points_shape: tuple, 
    safety_margin: float = 0.2,  # 20% safety margin
    device: str = "cuda"
) -> int:
    """Calculate safe batch size based on available GPU memory.
    
    Args:
        points_shape: Tuple of (batch_size, num_points, channels) or (batch_size, channels, num_points)
        safety_margin: Safety margin as fraction (0.0-1.0)
        device: Target device
    
    Returns:
        Maximum safe batch size (at least 1)
    """
    if device != "cuda" or not torch.cuda.is_available():
        return points_shape[0] if len(points_shape) >= 1 else 1
    
    # Parse input shape
    if len(points_shape) == 3:
        if points_shape[1] == 3:  # (B, 3, N)
            batch_size, channels, num_points = points_shape
        else:  # (B, N, 3)
            batch_size, num_points, channels = points_shape
    else:
        # Default fallback
        return max(1, points_shape[0] if len(points_shape) >= 1 else 1)
    
    # Check available memory
    total, allocated, free, free_percentage = check_gpu_memory_available(device)
    if free is None or free <= 0:
        return max(1, batch_size)
    
    # Calculate memory required per sample (rough estimation)
    # Each point has 3 coordinates (float32 = 4 bytes), plus intermediate activations
    # Approximate: points memory + feature memory + overhead
    points_memory_per_sample = num_points * channels * 4  # 4 bytes per float32
    feature_memory_per_sample = 1280 * 4  # 1280 features * 4 bytes (worst case)
    overhead_factor = 2.0  # Account for intermediate activations and gradients
    
    memory_per_sample = (points_memory_per_sample + feature_memory_per_sample) * overhead_factor
    
    # Apply safety margin to available memory
    safe_memory = free * (1.0 - safety_margin)
    
    # Calculate maximum batch size
    max_batch_size = int(safe_memory // memory_per_sample)
    
    return max(1, min(batch_size, max_batch_size))


class ULIPExtractor:
    """ULIP-2 feature extractor for point clouds.
    
    This class loads a pre-trained ULIP-2 model (or a dummy model as fallback)
    and extracts semantic features from point clouds.
    
    Args:
        checkpoint_path: Path to the pre-trained ULIP-2 checkpoint file (.pth)
        openclip_checkpoint: Optional path to local OpenCLIP checkpoint file (.bin)
            If provided and checkpoint_path exists, uses local weights instead of downloading
        device: Device to load the model on ('cuda' or 'cpu')
    
    Attributes:
        device: Device the model is loaded on
        use_dummy: Whether using dummy model (True) or real ULIP-2 (False)
        model: The loaded model (either real ULIP-2 or dummy)
        feature_dim: Feature dimension (1280 for real ULIP-2, 256 for dummy)
    """
    
    _lock = threading.Lock()  # class-level lock for monkey-patch thread safety
    
    def __init__(self, checkpoint_path: Optional[str] = None, 
                 openclip_checkpoint: Optional[str] = None,
                 device: str = "cuda"):
        """Initialize ULIP-2 extractor.
        
        Args:
            checkpoint_path: Path to checkpoint file, if None uses dummy model
            openclip_checkpoint: Optional path to local OpenCLIP checkpoint file.
                If provided and checkpoint_path exists, uses local weights instead of downloading.
                Should point to open_clip_pytorch_model.bin (9.5GB).
            device: Target device ('cuda' or 'cpu'), falls back to CPU if CUDA unavailable
        """
        self.device: str = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.use_dummy: bool = True
        self.model: nn.Module
        self.openclip_checkpoint = os.path.abspath(openclip_checkpoint) if openclip_checkpoint else None
        if self.openclip_checkpoint and not os.path.exists(self.openclip_checkpoint):
            logging.warning(f"OpenCLIP checkpoint not found: {self.openclip_checkpoint}, will download weights if needed")
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                # Add ULIP to path if not already
                ulip_path = os.path.join(os.path.dirname(__file__), '..', 'ULIP')
                if os.path.isdir(ulip_path):
                    if ulip_path not in sys.path:
                        sys.path.insert(0, ulip_path)

                
                # ULIP models will be imported after applying monkey-patches
                
                # Apply monkey-patch if openclip_checkpoint provided and exists
                with self._lock:
                    original_create = None
                    restore_pointnet2ops = None
                    restore_knn_cuda = None
                    
                    # Apply pointnet2_ops monkey-patch to avoid CUDA compilation
                    restore_pointnet2ops = _create_pointnet2ops_monkey_patch()
                    
                    # Apply knn_cuda monkey-patch to avoid CUDA compilation
                    restore_knn_cuda = _create_knn_cuda_monkey_patch()
                    
                    # Import ULIP models after patching
                    original_cwd = os.getcwd()
                    original_sys_path = sys.path.copy()
                    # Temporarily remove '.' and project root from sys.path to avoid namespace conflicts
                    project_root = os.path.abspath('.')
                    sys.path = [p for p in sys.path if p not in ('.', project_root)]
                    try:
                        os.chdir(ulip_path)
                        # Pre-import utils modules to avoid circular imports
                        logging.warning(f"sys.path: {sys.path}")
                        logging.warning(f"ulip_path: {ulip_path}")
                        logging.warning(f"utils/io.py exists: {os.path.exists(os.path.join(ulip_path, 'utils', 'io.py'))}")
                        logging.warning(f"utils in sys.modules: {'utils' in sys.modules}")
                        import utils.io
                        import utils.utils
                        import models.ULIP_models as ulip_models
                    finally:
                        os.chdir(original_cwd)
                        sys.path = original_sys_path
                    
                    if self.openclip_checkpoint and os.path.exists(self.openclip_checkpoint):
                        # Warn if file extension is unexpected
                        if not self.openclip_checkpoint.lower().endswith('.bin'):
                            logging.warning(f"OpenCLIP checkpoint file {self.openclip_checkpoint} does not have .bin extension")
                        import open_clip
                        original_create, patched_create = _create_openclip_monkey_patch(self.openclip_checkpoint)
                        open_clip.create_model_and_transforms = patched_create
                        logging.info(f"Using local OpenCLIP weights from {self.openclip_checkpoint}")
                    
                    try:
                        # Load checkpoint (weights_only=False for compatibility with numpy scalars)
                        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                        state_dict = ckpt['state_dict']
                        
                        # Remove 'module.' prefix from state dict keys
                        new_state_dict = {}
                        for k, v in state_dict.items():
                            if k.startswith('module.'):
                                new_state_dict[k[7:]] = v
                            else:
                                new_state_dict[k] = v
                        state_dict = new_state_dict
                        
                        # Determine model architecture based on checkpoint name or default
                        # Currently hardcoded for ULIP-2 PointBERT colored model

                        model_name = 'ULIP2_PointBERT_Colored'
                        
                        # Create model with dummy args
                        class Args:
                            evaluate_3d = True
                            use_height = False
                            npoints = 8192
                        
                        args = Args()
                        
                        # Create model instance (uses patched open_clip.create_model_and_transforms if applied)
                        original_cwd = os.getcwd()
                        try:
                            os.chdir(ulip_path)
                            model = getattr(ulip_models, model_name)(args)
                        finally:
                            os.chdir(original_cwd)
                        model.eval()
                        model.to(self.device)
                        
                        # Load state dict
                        model.load_state_dict(state_dict, strict=False)
                        
                        self.model = model
                        self.use_dummy = False
                        logging.info(f"ULIP-2 model loaded successfully from {checkpoint_path}, feature_dim={self.feature_dim}")
                        
                    finally:
                        # Restore original function if patched
                        if original_create is not None:
                            import open_clip
                            open_clip.create_model_and_transforms = original_create
                
            except Exception as e:
                logging.warning(f"ULIP-2 loading failed, using dummy model: {e}")
                import traceback
                logging.warning(traceback.format_exc())
                self._init_dummy_model()
        else:
            self._init_dummy_model()
    
    @property
    def feature_dim(self) -> int:
        """Return feature dimension (1280 for real ULIP-2, 256 for dummy)."""
        return 1280 if not self.use_dummy else 256
    
    def _init_dummy_model(self) -> None:
        """Initialize dummy ULIP-2 model with per-point MLP and global pooling.
        
        The dummy model processes each point independently through a small MLP
        (3→128→256) and applies global average pooling across points.
        """
        class DummyULIP(nn.Module):
            def __init__(self, output_dim=256):
                super().__init__()
                self.point_encoder = nn.Sequential(
                    nn.Linear(3, 128),
                    nn.ReLU(),
                    nn.Linear(128, output_dim)
                )
            
            def forward(self, points: torch.Tensor) -> torch.Tensor:
                """Forward pass of dummy ULIP model.
                
                Args:
                    points: Point cloud tensor of shape (B, N, 3)
                
                Returns:
                    Global features tensor of shape (B, 256)
                """
                batch_size, num_points, _ = points.shape
                points_flat = points.reshape(-1, 3)
                features_flat = self.point_encoder(points_flat)
                features = features_flat.reshape(batch_size, num_points, -1)
                global_feature = features.mean(dim=1)
                return global_feature
        
        self.dummy_model = DummyULIP(output_dim=256)
        self.dummy_model.eval()
        self.dummy_model.to(self.device)
        self.model = self.dummy_model
        logging.info(f"Initialized dummy ULIP model with feature_dim={self.feature_dim}")
    
    def extract_features(self, point_clouds: torch.Tensor) -> torch.Tensor:
        """Extract semantic features from point clouds.
        
        Args:
            point_clouds: Point cloud tensor of shape (B, N, 3) or (B, 3, N)
        
        Returns:
            Semantic features tensor of shape (B, feature_dim) where feature_dim is 256 for dummy model or 1280 for real ULIP-2
        
        Raises:
            RuntimeError: If feature extraction fails
        """
        # Validate input dimensions
        if point_clouds.dim() != 3:
            raise RuntimeError(f"Points tensor must be 3D (B, N, 3) or (B, 3, N), got shape {point_clouds.shape}")
        
        # Determine input format and convert to (B, N, 3) if needed
        if point_clouds.shape[1] == 3 and point_clouds.shape[2] != 3:
            # (B, 3, N) format, convert to (B, N, 3)
            point_clouds = point_clouds.transpose(1, 2).contiguous()
        elif point_clouds.shape[2] == 3:
            # (B, N, 3) format, keep as is
            pass
        else:
            raise RuntimeError(
                f"Invalid point cloud shape {point_clouds.shape}. "
                f"Expected channels=3 in dim 1 or 2."
            )
        
        with torch.no_grad():
            point_clouds = point_clouds.to(self.device)
            if self.use_dummy:
                return self.model(point_clouds).cpu()
            else:
                # ULIP2 model has encode_pc method
                # For colored model, ensure 6 channels (xyz + rgb)
                if point_clouds.shape[2] == 3:
                    # Pad with zeros for RGB channels
                    zeros = torch.zeros_like(point_clouds[..., :3])
                    point_clouds = torch.cat([point_clouds, zeros], dim=2)
                features = self.model.encode_pc(point_clouds)
                return features.cpu()