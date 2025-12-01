"""
Scene Explorer - Generating multiple viewpoints for scene analysis
Stage 2: Scene Exploration
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
from scipy.spatial.transform import Rotation


class SceneExplorer:
    """Viewpoint generation via Spherical Grid Sampling"""
    
    def __init__(self, center: np.ndarray, radius: float, config: Dict):
        """
        Initialize scene explorer
        
        Args:
            center: Scene center (x, y, z)
            radius: Scene radius
            config: Configuration from config.yaml (scene_explorer section)
        """
        self.center = np.array(center)
        self.radius = radius
        self.config = config
        
        self.viewpoints = []
    
    def generate_viewpoints(self) -> List[Dict]:
        """
        Generate viewpoints via spherical grid
        
        Returns:
            List[Dict]: List of viewpoints with position and lookAt
        """
        print("Generating viewpoints using spherical grid sampling...")
        
        num_radii = self.config.get('num_radii', 5)
        num_angles = self.config.get('num_angles', 20)
        num_heights = self.config.get('num_heights', 5)
        min_radius_factor = self.config.get('min_radius_factor', 0.5)
        max_radius_factor = self.config.get('max_radius_factor', 2.0)
        
        min_radius = self.radius * min_radius_factor
        max_radius = self.radius * max_radius_factor
        
        # Generate radii
        radii = np.linspace(min_radius, max_radius, num_radii)
        
        # Generate angles (azimuth)
        angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
        
        # Generate heights (relative to center)
        height_range = self.radius * 0.8  # Height varies within 80% of radius
        heights = np.linspace(-height_range, height_range, num_heights)
        
        viewpoints = []
        
        for r in radii:
            for angle in angles:
                for height in heights:
                    # Compute position on sphere
                    x = r * np.cos(angle)
                    z = r * np.sin(angle)
                    y = height
                    
                    position = self.center + np.array([x, y, z])
                    
                    # Look direction toward scene center
                    direction = self.center - position
                    direction_norm = np.linalg.norm(direction)
                    
                    if direction_norm > 0.01:  # Avoid division by zero
                        direction = direction / direction_norm
                        look_at = position + direction
                    else:
                        look_at = position + np.array([0, 0, -1])
                    
                    viewpoints.append({
                        'position': position.tolist(),
                        'lookAt': look_at.tolist()
                    })
        
        self.viewpoints = viewpoints
        print(f"Generated {len(viewpoints)} viewpoints")
        
        return viewpoints
    
    def save_viewpoints(self, output_path: str):
        """
        Save viewpoints to JSON
        
        Args:
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'center': self.center.tolist(),
                'radius': self.radius,
                'viewpoints': self.viewpoints
            }, f, indent=2)
        
        print(f"Saved viewpoints to {output_path}")
    
    def load_viewpoints(self, input_path: str) -> List[Dict]:
        """
        Load viewpoints from JSON
        
        Args:
            input_path: Path to JSON file
            
        Returns:
            List[Dict]: List of viewpoints
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        self.viewpoints = data['viewpoints']
        return self.viewpoints


if __name__ == "__main__":
    # Testing
    center = np.array([0, 0, 0])
    radius = 10.0
    config = {
        'num_radii': 3,
        'num_angles': 8,
        'num_heights': 3
    }
    
    explorer = SceneExplorer(center, radius, config)
    viewpoints = explorer.generate_viewpoints()
    print(f"Generated {len(viewpoints)} viewpoints")
    print(f"First viewpoint: {viewpoints[0]}")

