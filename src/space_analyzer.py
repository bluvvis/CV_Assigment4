"""
Space Analyzer - Finding free space and walkable paths
Stage 3: Space Analysis
"""

import numpy as np
from typing import List, Tuple, Dict
from scipy.spatial import cKDTree


class SpaceAnalyzer:
    """Space analysis via voxel grid"""
    
    def __init__(self, points: np.ndarray, config: Dict):
        """
        Initialize space analyzer
        
        Args:
            points: Point array (N×3)
            config: Configuration from config.yaml (space_analyzer section)
        """
        self.points = points
        self.config = config
        
        self.voxel_size = config.get('voxel_size', 1.0)
        self.density_threshold = config.get('density_threshold', 10)
        self.floor_percentile = config.get('floor_percentile', 10)
        self.camera_height = config.get('camera_height', 1.7)
        self.camera_height_tolerance = config.get('camera_height_tolerance', 0.2)
        
        self.voxel_grid = None
        self.floor_level = None
        self.walkable_cells = []
    
    def analyze(self) -> List[Tuple[int, int, int]]:
        """
        Analyze space and find walkable paths
        
        Returns:
            List[Tuple[int, int, int]]: List of walkable cells (voxel indices)
        """
        print("Analyzing space...")
        
        # Check scene size - use simplified mode for very large scenes
        num_points = len(self.points)
        if num_points > 10_000_000:
            print(f"  Large scene detected ({num_points:,} points), using simplified analysis...")
            return self._simplified_analysis()
        
        # 1. Create voxel grid
        self._create_voxel_grid()
        
        # 2. Floor detection
        self._detect_floor()
        
        # 3. Find free space
        free_voxels = self._find_free_space()
        
        # 4. Extract walkable paths
        self.walkable_cells = self._extract_walkable_cells(free_voxels)
        
        print(f"Found {len(self.walkable_cells)} walkable cells")
        
        return self.walkable_cells
    
    def _simplified_analysis(self) -> List[Tuple[int, int, int]]:
        """Simplified analysis for very large scenes"""
        print("  Using simplified analysis (sampling-based)...")
        
        # Floor detection
        self._detect_floor()
        
        # IMPORTANT: Use FULL bounds of entire scene, not sample bounds!
        min_bounds = np.min(self.points, axis=0)
        max_bounds = np.max(self.points, axis=0)
        center = (min_bounds + max_bounds) / 2.0
        
        # Increase voxel_size for simplification
        simplified_voxel_size = self.voxel_size * 5.0
        
        # Generate walkable cells based on real scene points
        camera_y = self.floor_level + self.camera_height
        
        # Use sampling for fast search of points at camera height
        sample_size = min(500_000, len(self.points))
        sample_indices = np.random.choice(len(self.points), sample_size, replace=False)
        sample_points = self.points[sample_indices]
        
        # Find points at camera height (within tolerance)
        height_mask = np.abs(sample_points[:, 1] - camera_y) < self.camera_height_tolerance * 3
        points_at_height = sample_points[height_mask]
        
        if len(points_at_height) == 0:
            # If no points at required height, use median height
            camera_y = np.median(sample_points[:, 1])
            height_mask = np.abs(sample_points[:, 1] - camera_y) < self.camera_height_tolerance * 5
            points_at_height = sample_points[height_mask]
        
        print(f"  Found {len(points_at_height)} points at camera height")
        
        # Create KD-tree for fast search
        from scipy.spatial import cKDTree
        kdtree = cKDTree(points_at_height)
        
        # Generate grid of walkable positions in central region (80% of bounds)
        inner_min = min_bounds + (max_bounds - min_bounds) * 0.1
        inner_max = min_bounds + (max_bounds - min_bounds) * 0.9
        
        # Create grid in central region
        num_grid_x = 30
        num_grid_z = 30
        x_range = np.linspace(inner_min[0], inner_max[0], num_grid_x)
        z_range = np.linspace(inner_min[2], inner_max[2], num_grid_z)
        
        walkable_cells = []
        search_radius = simplified_voxel_size * 2.0
        
        for x in x_range:
            for z in z_range:
                # Find nearest points at camera height
                query_point = np.array([x, camera_y, z])
                distances, indices = kdtree.query(query_point, k=5)
                
                # Check that there are points nearby (but not too close - this is free space)
                nearby_count = np.sum(distances < search_radius)
                far_count = np.sum((distances > search_radius * 0.5) & (distances < search_radius * 2))
                
                # Walkable position: has points nearby (structure), but not too dense (free space)
                if 5 <= nearby_count <= 50 and far_count > 0:
                    # Convert to voxel indices
                    i = int((x - min_bounds[0]) / simplified_voxel_size)
                    j = int((camera_y - min_bounds[1]) / simplified_voxel_size)
                    k = int((z - min_bounds[2]) / simplified_voxel_size)
                    walkable_cells.append((i, j, k))
        
        self.walkable_cells = walkable_cells[:1000]  # Limit quantity
        self.min_bounds = min_bounds
        self.voxel_size = simplified_voxel_size
        
        print(f"  Found {len(self.walkable_cells)} walkable cells (simplified)")
        return self.walkable_cells
    
    def _create_voxel_grid(self):
        """Create voxel grid and compute density"""
        print("Creating voxel grid...")
        
        # Scene bounds
        min_bounds = np.min(self.points, axis=0)
        max_bounds = np.max(self.points, axis=0)
        
        # Grid dimensions
        grid_size = ((max_bounds - min_bounds) / self.voxel_size).astype(int) + 1
        
        # Check grid size - if too large, increase voxel_size
        total_voxels = np.prod(grid_size)
        if total_voxels > 100_000_000:  # If more than 100M voxels
            print(f"  Warning: Grid too large ({total_voxels:,} voxels), increasing voxel_size...")
            # Increase voxel_size to reduce grid
            scale_factor = (total_voxels / 50_000_000) ** (1/3)  # Target size ~50M
            self.voxel_size *= scale_factor
            grid_size = ((max_bounds - min_bounds) / self.voxel_size).astype(int) + 1
            print(f"  Adjusted voxel_size to {self.voxel_size:.2f}m")
        
        # Create grid
        self.voxel_grid = np.zeros(grid_size, dtype=np.int32)
        self.min_bounds = min_bounds
        self.grid_size = grid_size
        
        # Optimization: use numpy for fast counting
        print("  Computing voxel densities (this may take a moment)...")
        voxel_indices = ((self.points - min_bounds) / self.voxel_size).astype(int)
        
        # Clip indices to grid boundaries
        voxel_indices = np.clip(voxel_indices, 0, np.array(grid_size) - 1)
        
        # Fast counting via bincount (for 1D) or via unique
        # Use more efficient method
        flat_indices = np.ravel_multi_index(voxel_indices.T, grid_size)
        unique, counts = np.unique(flat_indices, return_counts=True)
        self.voxel_grid.flat[unique] = counts
        
        print(f"Voxel grid size: {grid_size}")
        print(f"Total voxels: {np.prod(grid_size):,}")
    
    def _detect_floor(self):
        """Detect floor level via Y-coordinate percentile"""
        y_coords = self.points[:, 1]  # Y coordinates
        self.floor_level = np.percentile(y_coords, self.floor_percentile)
        print(f"Floor level detected at Y = {self.floor_level:.2f}")
    
    def _find_free_space(self) -> List[Tuple[int, int, int]]:
        """Find free space (voxels with low density)"""
        print("Finding free space...")
        
        # Optimization: use numpy for fast search
        # Find all voxels with low density in one pass
        free_mask = self.voxel_grid <= self.density_threshold
        free_indices = np.argwhere(free_mask)
        
        # Convert to list of tuples
        free_voxels = [tuple(idx) for idx in free_indices]
        
        print(f"Found {len(free_voxels)} free voxels")
        return free_voxels
    
    def _extract_walkable_cells(self, free_voxels: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """
        Extract walkable cells at camera height
        
        Args:
            free_voxels: List of free voxels
            
        Returns:
            List[Tuple[int, int, int]]: Walkable cells
        """
        print("Extracting walkable cells...")
        
        # Camera height above floor
        camera_y = self.floor_level + self.camera_height
        
        # Height range for camera
        min_y = camera_y - self.camera_height_tolerance
        max_y = camera_y + self.camera_height_tolerance
        
        # Convert to voxel indices
        min_y_idx = int((min_y - self.min_bounds[1]) / self.voxel_size)
        max_y_idx = int((max_y - self.min_bounds[1]) / self.voxel_size)
        
        walkable_cells = []
        
        for i, j, k in free_voxels:
            # Check height (j corresponds to Y in our grid)
            if min_y_idx <= j <= max_y_idx:
                # Check that cell is surrounded by structure (has neighbors with high density)
                if self._is_surrounded_by_structure(i, j, k):
                    walkable_cells.append((i, j, k))
        
        return walkable_cells
    
    def _is_surrounded_by_structure(self, i: int, j: int, k: int) -> bool:
        """
        Check that cell is surrounded by structure (walls/objects)
        
        Args:
            i, j, k: Voxel indices
            
        Returns:
            bool: True if surrounded by structure
        """
        # Check neighbors in XZ plane (horizontal)
        neighbors = [
            (i-1, j, k), (i+1, j, k),  # Neighbors along X
            (i, j, k-1), (i, j, k+1),  # Neighbors along Z
        ]
        
        structure_count = 0
        for ni, nj, nk in neighbors:
            if (0 <= ni < self.grid_size[0] and 
                0 <= nj < self.grid_size[1] and 
                0 <= nk < self.grid_size[2]):
                if self.voxel_grid[ni, nj, nk] > self.density_threshold:
                    structure_count += 1
        
        # Require at least 2 neighbors with structure
        return structure_count >= 2
    
    def voxel_to_world(self, voxel_idx: Tuple[int, int, int]) -> np.ndarray:
        """
        Convert voxel index to world coordinates
        
        Args:
            voxel_idx: Voxel index (i, j, k)
            
        Returns:
            np.ndarray: World coordinates (x, y, z)
        """
        i, j, k = voxel_idx
        world_pos = self.min_bounds + np.array([i, j, k]) * self.voxel_size
        return world_pos
    
    def get_walkable_positions(self) -> List[np.ndarray]:
        """Get world coordinates of walkable cells"""
        return [self.voxel_to_world(cell) for cell in self.walkable_cells]


if __name__ == "__main__":
    # Testing
    np.random.seed(42)
    points = np.random.rand(1000, 3) * 10
    
    config = {
        'voxel_size': 1.0,
        'density_threshold': 10,
        'floor_percentile': 10,
        'camera_height': 1.7,
        'camera_height_tolerance': 0.2
    }
    
    analyzer = SpaceAnalyzer(points, config)
    walkable = analyzer.analyze()
    print(f"Found {len(walkable)} walkable cells")

