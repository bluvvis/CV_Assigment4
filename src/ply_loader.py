"""
PLY Loader - Loading and parsing 3D Gaussian Splatting data
Stage 1: Scene Loading
"""

import numpy as np
from plyfile import PlyData
from pathlib import Path
from typing import Tuple, Dict, Optional


class PLYLoader:
    """Loading and parsing PLY files for 3D Gaussian Splatting"""
    
    def __init__(self, ply_path: str):
        """
        Initialize loader
        
        Args:
            ply_path: Path to PLY or .splat file
        """
        self.ply_path = Path(ply_path)
        if not self.ply_path.exists():
            raise FileNotFoundError(f"File not found: {ply_path}")
        
        # Check file extension
        if self.ply_path.suffix.lower() == '.splat':
            # .splat files can be PLY files with different extension
            # or binary format - try loading as PLY first
            pass
        
        self.points = None
        self.metadata = {}
    
    def load(self) -> Tuple[np.ndarray, Dict]:
        """
        Load PLY file
        
        Returns:
            Tuple[np.ndarray, Dict]: Point array (N×3) and scene metadata
        """
        print(f"Loading file: {self.ply_path}")
        
        # Try loading as PLY
        try:
            # Load PLY data
            plydata = PlyData.read(str(self.ply_path))
            
            # Extract point coordinates
            # In Gaussian Splatting, usually x, y, z fields are used
            vertex_data = plydata['vertex']
            
            # Get coordinates
            x = vertex_data['x']
            y = vertex_data['y']
            z = vertex_data['z']
            
            # Combine into point array (N×3)
            self.points = np.column_stack([x, y, z])
            
        except Exception as e:
            # If not PLY, try loading as binary .splat
            if self.ply_path.suffix.lower() == '.splat':
                print("Detected binary .splat format, loading...")
                self.points = self._load_binary_splat()
            else:
                raise
        
        print(f"Loaded {len(self.points)} points")
        
        # Compute metadata
        self.metadata = self._compute_metadata()
        
        return self.points, self.metadata
    
    def _compute_metadata(self) -> Dict:
        """
        Compute scene metadata
        
        Returns:
            Dict: Metadata (bounds, center, radius)
        """
        if self.points is None or len(self.points) == 0:
            return {}
        
        # Scene bounds
        min_bounds = np.min(self.points, axis=0)
        max_bounds = np.max(self.points, axis=0)
        
        # Scene center
        center = (min_bounds + max_bounds) / 2.0
        
        # Scene size
        size = max_bounds - min_bounds
        
        # Scene radius (maximum distance from center to corner)
        radius = np.linalg.norm(size) / 2.0
        
        metadata = {
            'min_bounds': min_bounds.tolist(),
            'max_bounds': max_bounds.tolist(),
            'center': center.tolist(),
            'size': size.tolist(),
            'radius': float(radius),
            'num_points': len(self.points)
        }
        
        print(f"Scene metadata:")
        print(f"  Center: {center}")
        print(f"  Size: {size}")
        print(f"  Radius: {radius:.2f}m")
        
        return metadata
    
    def get_points(self) -> Optional[np.ndarray]:
        """Get loaded points"""
        return self.points
    
    def get_metadata(self) -> Dict:
        """Get scene metadata"""
        return self.metadata
    
    def _load_binary_splat(self) -> np.ndarray:
        """
        Load binary .splat format
        
        Format: each Gaussian splat is represented by 32-byte structure:
        - x, y, z: 3 * float32 (12 bytes) - position
        - rotation: 4 * float32 (16 bytes) - quaternion
        - scale: 3 * float32 (12 bytes) - but that's already 40 bytes...
        - opacity: float32 (4 bytes)
        - color: 3 * uint8 (3 bytes)
        - padding: 1 byte
        
        In practice, 32-byte structure is used, where:
        - x, y, z: first 12 bytes (3 * float32)
        - remaining 20 bytes: rotation, scale, opacity, color in compact format
        
        Returns:
            np.ndarray: Point array (N×3)
        """
        with open(self.ply_path, 'rb') as f:
            data = f.read()
        
        # Size of one splat structure (32 bytes)
        SPLAT_SIZE = 32
        num_splats = len(data) // SPLAT_SIZE
        
        if num_splats == 0:
            raise ValueError(f"File {self.ply_path} is too small or has wrong format")
        
        print(f"Loading {num_splats:,} Gaussian splats from binary format...")
        
        # Extract coordinates (first 12 bytes of each splat = 3 * float32)
        # Read all data as float32 and take every 8th value (32 bytes / 4 = 8 float32 per splat)
        all_floats = np.frombuffer(data[:num_splats * SPLAT_SIZE], dtype=np.float32)
        
        # Reshape: every 8 float32 = one splat, take first 3 (x, y, z)
        points = all_floats.reshape(-1, 8)[:, :3]
        
        print(f"Extracted {len(points):,} points from binary splat file")
        
        return points


if __name__ == "__main__":
    # Test loader
    import sys
    
    if len(sys.argv) > 1:
        loader = PLYLoader(sys.argv[1])
        points, metadata = loader.load()
        print(f"\nLoaded {len(points)} points")
        print(f"Metadata: {metadata}")
    else:
        print("Usage: python ply_loader.py <path_to_ply_file>")

