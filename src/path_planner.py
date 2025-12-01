"""
Path Planner - Generating smooth camera trajectories
Stage 4: Path Planning
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d


class PathPlanner:
    """Camera trajectory planning (Flythrough and Orbital)"""
    
    def __init__(self, points: np.ndarray, walkable_cells: List[Tuple[int, int, int]], 
                 space_analyzer, config: Dict):
        """
        Initialize path planner
        
        Args:
            points: Scene point array (N×3)
            walkable_cells: List of walkable cells
            space_analyzer: SpaceAnalyzer instance for coordinate transformation
            config: Configuration from config.yaml (path_planner section)
        """
        self.points = points
        self.walkable_cells = walkable_cells
        self.space_analyzer = space_analyzer
        self.config = config
        
        # KD-tree for collision checking
        self.kdtree = cKDTree(points)
    
    def plan_flythrough(self, start_pos: Optional[np.ndarray] = None, 
                       end_pos: Optional[np.ndarray] = None,
                       num_keyframes: int = 100) -> List[Dict]:
        """
        Plan flythrough path (for interiors)
        
        Args:
            start_pos: Start position (if None, selected automatically)
            end_pos: End position (if None, selected automatically)
            num_keyframes: Number of keyframes
            
        Returns:
            List[Dict]: List of keyframes with position, lookAt, frame
        """
        print("Planning flythrough path...")
        
        # Compute scene bounds to constrain path
        min_bounds = np.min(self.points, axis=0)
        max_bounds = np.max(self.points, axis=0)
        center = (min_bounds + max_bounds) / 2.0
        
        # Get walkable positions
        walkable_positions = self.space_analyzer.get_walkable_positions()
        
        # Filter: keep only positions inside scene bounds
        # and close to scene points (within 2x voxel_size)
        filtered_positions = []
        voxel_size = self.space_analyzer.voxel_size
        max_distance = voxel_size * 2.0
        
        for pos in walkable_positions:
            # Check that position is inside bounds
            if not (np.all(pos >= min_bounds - voxel_size) and np.all(pos <= max_bounds + voxel_size)):
                continue
            
            # Check that position is close to some scene point
            distances, _ = self.kdtree.query(pos, k=1)
            if distances < max_distance:
                filtered_positions.append(pos)
        
        if len(filtered_positions) == 0:
            print("Warning: No valid walkable cells found, using path based on scene points")
            return self._create_path_from_scene_points(num_keyframes, center, min_bounds, max_bounds)
        
        walkable_positions = filtered_positions
        print(f"Using {len(walkable_positions)} walkable positions (filtered from scene bounds)")
        
        # Select start and end positions inside room
        if start_pos is None:
            # Select position closest to scene center (in central region)
            center_region_radius = np.linalg.norm(max_bounds - min_bounds) * 0.3  # 30% of size
            positions_near_center = [
                pos for pos in walkable_positions 
                if np.linalg.norm(pos - center) < center_region_radius
            ]
            
            if len(positions_near_center) > 0:
                # Select from positions close to center
                distances_to_center = [np.linalg.norm(pos - center) for pos in positions_near_center]
                start_idx = np.argmin(distances_to_center)
                start_pos = positions_near_center[start_idx]
            else:
                # Fallback: closest to center from all
                distances_to_center = [np.linalg.norm(pos - center) for pos in walkable_positions]
                start_idx = np.argmin(distances_to_center)
                start_pos = walkable_positions[start_idx]
        
        if end_pos is None:
            # Select position far from start, but also in central region
            center_region_radius = np.linalg.norm(max_bounds - min_bounds) * 0.4  # 40% of size
            positions_near_center = [
                pos for pos in walkable_positions 
                if np.linalg.norm(pos - center) < center_region_radius
            ]
            
            if len(positions_near_center) > 1:
                # Select position maximally far from start, but in central region
                distances_from_start = [np.linalg.norm(pos - start_pos) for pos in positions_near_center]
                end_idx = np.argmax(distances_from_start)
                end_pos = positions_near_center[end_idx]
            else:
                # Fallback: position at 70% of maximum distance
                distances_from_start = [np.linalg.norm(pos - start_pos) for pos in walkable_positions]
                max_dist = max(distances_from_start)
                target_dist = max_dist * 0.7
                end_idx = np.argmin([abs(d - target_dist) for d in distances_from_start])
                end_pos = walkable_positions[end_idx]
        
        # A* on 2D XZ projection
        waypoints = self._a_star_2d(walkable_positions, start_pos, end_pos)
        
        if len(waypoints) < 2:
            waypoints = [start_pos, end_pos]
        
        # Filter waypoints - remove those far from scene points
        filtered_waypoints = []
        for wp in waypoints:
            distances, _ = self.kdtree.query(wp, k=1)
            if distances < max_distance * 1.5:  # Slightly larger tolerance for waypoints
                filtered_waypoints.append(wp)
        
        if len(filtered_waypoints) < 2:
            filtered_waypoints = [start_pos, end_pos]
        
        # Interpolation between waypoints
        keyframes = self._interpolate_waypoints(filtered_waypoints, num_keyframes)
        
        # Check and correct keyframes - ensure they are inside room
        keyframes = self._constrain_keyframes_to_scene(keyframes, min_bounds, max_bounds)
        
        print(f"Generated {len(keyframes)} keyframes for flythrough")
        
        return keyframes
    
    def plan_coverage(self, num_keyframes: int = 500) -> List[Dict]:
        """
        Plan path for covering most of the scene
        
        Args:
            num_keyframes: Number of keyframes
            
        Returns:
            List[Dict]: List of keyframes covering most of the scene
        """
        print("Planning coverage path (covering most of the scene)...")
        
        # Compute scene bounds
        min_bounds = np.min(self.points, axis=0)
        max_bounds = np.max(self.points, axis=0)
        center = (min_bounds + max_bounds) / 2.0
        scene_size = max_bounds - min_bounds
        
        # Get walkable positions
        walkable_positions = self.space_analyzer.get_walkable_positions()
        
        # Filter positions inside bounds
        voxel_size = self.space_analyzer.voxel_size
        max_distance = voxel_size * 2.0
        
        filtered_positions = []
        for pos in walkable_positions:
            if not (np.all(pos >= min_bounds - voxel_size) and np.all(pos <= max_bounds + voxel_size)):
                continue
            distances, _ = self.kdtree.query(pos, k=1)
            if distances < max_distance:
                filtered_positions.append(pos)
        
        if len(filtered_positions) < 4:
            print("Warning: Not enough walkable positions, using simple flythrough")
            return self.plan_flythrough(num_keyframes=num_keyframes)
        
        walkable_positions = np.array(filtered_positions)
        print(f"Using {len(walkable_positions)} walkable positions for coverage")
        
        # Divide scene into regions (grid) and select key points only in regions with high density
        # Use 3D grid for better coverage
        grid_size = 3  # 3x3x2 = 18 regions (fewer for smoother path)
        key_points = []
        
        # Compute point density in each cell
        cell_size = scene_size / grid_size
        min_points_per_cell = len(self.points) / (grid_size * grid_size * 2) * 0.3  # Minimum 30% of average density
        
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(2):  # 2 height levels
                    # Current grid cell boundaries
                    cell_min = min_bounds + scene_size * np.array([
                        i / grid_size,
                        k * 0.4 + 0.1,  # Lower and upper levels (10%-50% and 50%-90%)
                        j / grid_size
                    ])
                    cell_max = min_bounds + scene_size * np.array([
                        (i + 1) / grid_size,
                        (k + 1) * 0.4 + 0.1,
                        (j + 1) / grid_size
                    ])
                    cell_center = (cell_min + cell_max) / 2.0
                    
                    # Check point density in this cell
                    points_in_cell = self.points[
                        (self.points[:, 0] >= cell_min[0]) & (self.points[:, 0] < cell_max[0]) &
                        (self.points[:, 1] >= cell_min[1]) & (self.points[:, 1] < cell_max[1]) &
                        (self.points[:, 2] >= cell_min[2]) & (self.points[:, 2] < cell_max[2])
                    ]
                    
                    # Skip empty cells
                    if len(points_in_cell) < min_points_per_cell:
                        continue
                    
                    # Find nearest walkable position in this cell
                    if len(walkable_positions) > 0:
                        # Filter walkable positions within cell
                        in_cell_mask = (
                            (walkable_positions[:, 0] >= cell_min[0] - cell_size[0] * 0.2) &
                            (walkable_positions[:, 0] < cell_max[0] + cell_size[0] * 0.2) &
                            (walkable_positions[:, 1] >= cell_min[1] - cell_size[1] * 0.2) &
                            (walkable_positions[:, 1] < cell_max[1] + cell_size[1] * 0.2) &
                            (walkable_positions[:, 2] >= cell_min[2] - cell_size[2] * 0.2) &
                            (walkable_positions[:, 2] < cell_max[2] + cell_size[2] * 0.2)
                        )
                        positions_in_cell = walkable_positions[in_cell_mask]
                        
                        if len(positions_in_cell) > 0:
                            # Select position closest to cell center
                            distances = np.linalg.norm(positions_in_cell - cell_center, axis=1)
                            closest_idx = np.argmin(distances)
                            if distances[closest_idx] < np.linalg.norm(cell_size) * 0.5:  # Within cell
                                key_points.append(positions_in_cell[closest_idx])
        
        # Remove duplicates (close points) and sort by distance from center
        unique_key_points = []
        min_distance_between_points = np.linalg.norm(scene_size) * 0.15  # Minimum 15% of scene size
        
        for point in key_points:
            is_unique = True
            for existing in unique_key_points:
                if np.linalg.norm(point - existing) < min_distance_between_points:
                    is_unique = False
                    break
            if is_unique:
                unique_key_points.append(point)
        
        if len(unique_key_points) < 3:
            print("Warning: Not enough key points found, using flythrough with more waypoints")
            # Fallback: use flythrough, but with more intermediate points
            return self.plan_flythrough(num_keyframes=num_keyframes)
        
        print(f"Selected {len(unique_key_points)} key points for coverage")
        
        # Build path through all key points using A* between adjacent points
        # First order points (greedy algorithm with improvement)
        if len(unique_key_points) == 0:
            return self.plan_flythrough(num_keyframes=num_keyframes)
        
        # Start with point closest to center
        unique_key_points = np.array(unique_key_points)
        center_distances = np.linalg.norm(unique_key_points - center, axis=1)
        current_idx = np.argmin(center_distances)
        path_points = [unique_key_points[current_idx]]
        remaining_indices = set(range(len(unique_key_points))) - {current_idx}
        
        # Improved greedy algorithm: consider movement direction
        while remaining_indices:
            current_point = path_points[-1]
            min_score = float('inf')
            next_idx = None
            
            for idx in remaining_indices:
                candidate = unique_key_points[idx]
                dist = np.linalg.norm(candidate - current_point)
                
                # Penalty for too large jumps
                if dist > np.linalg.norm(scene_size) * 0.4:
                    continue
                
                # Bonus for movement in same direction
                if len(path_points) > 1:
                    prev_direction = path_points[-1] - path_points[-2]
                    new_direction = candidate - current_point
                    if np.linalg.norm(prev_direction) > 0.01 and np.linalg.norm(new_direction) > 0.01:
                        prev_direction = prev_direction / np.linalg.norm(prev_direction)
                        new_direction = new_direction / np.linalg.norm(new_direction)
                        direction_similarity = np.dot(prev_direction, new_direction)
                        # Penalty for sharp turns
                        score = dist * (2.0 - direction_similarity)
                    else:
                        score = dist
                else:
                    score = dist
                
                if score < min_score:
                    min_score = score
                    next_idx = idx
            
            if next_idx is not None:
                path_points.append(unique_key_points[next_idx])
                remaining_indices.remove(next_idx)
            else:
                # If no suitable point found, take nearest
                if remaining_indices:
                    distances = [np.linalg.norm(unique_key_points[idx] - current_point) 
                                for idx in remaining_indices]
                    next_idx = list(remaining_indices)[np.argmin(distances)]
                    path_points.append(unique_key_points[next_idx])
                    remaining_indices.remove(next_idx)
        
        # Use A* to build smooth path between key points
        waypoints = []
        for i in range(len(path_points) - 1):
            start = path_points[i]
            end = path_points[i + 1]
            
            # Use A* to build path between points
            # Convert walkable_positions to list of numpy arrays
            if isinstance(walkable_positions, np.ndarray):
                walkable_list = [walkable_positions[i] for i in range(len(walkable_positions))]
            else:
                walkable_list = walkable_positions
            
            segment_waypoints = self._a_star_2d(walkable_list, start, end)
            
            # Add waypoints (except last to avoid duplicates)
            if i == 0:
                waypoints.extend(segment_waypoints)
            else:
                waypoints.extend(segment_waypoints[1:])  # Skip first (already exists)
        
        # If too few waypoints, add intermediate points
        if len(waypoints) < 5:
            waypoints = path_points.tolist()
        
        # Use smoothing via Catmull-Rom splines
        keyframes = self._interpolate_waypoints(waypoints, num_keyframes)
        
        # Improve lookAt for each frame
        for i, kf in enumerate(keyframes):
            pos = np.array(kf['position'])
            
            # LookAt: look forward along path
            if i < len(keyframes) - 1:
                next_pos = np.array(keyframes[i + 1]['position'])
                direction = next_pos - pos
                if np.linalg.norm(direction) > 0.01:
                    direction = direction / np.linalg.norm(direction)
                    look_at = pos + direction * 5.0  # Look 5 units ahead
                else:
                    look_at = center
            else:
                # Last frame looks at center
                look_at = center
            
            kf['lookAt'] = look_at.tolist()
        
        # Constrain keyframes to scene bounds
        keyframes = self._constrain_keyframes_to_scene(keyframes, min_bounds, max_bounds)
        
        print(f"Generated {len(keyframes)} keyframes for coverage path")
        return keyframes
    
    def plan_orbital(self, center: np.ndarray, radius: float, 
                    num_keyframes: int = 100) -> List[Dict]:
        """
        Планирование орбитального пути (для экстерьеров)
        
        Args:
            center: Центр орбиты
            radius: Радиус орбиты
            num_keyframes: Количество ключевых кадров
            
        Returns:
            List[Dict]: Список ключевых кадров
        """
        print("Planning orbital path...")
        
        orbital_config = self.config.get('orbital', {})
        num_orbits = orbital_config.get('num_orbits', 3)
        spiral_height_range = orbital_config.get('spiral_height_range', 5.0)
        
        keyframes = []
        
        for frame_idx in range(num_keyframes):
            # Угол орбиты
            angle = (frame_idx / num_keyframes) * 2 * np.pi * num_orbits
            
            # Высота (спираль)
            height_factor = (frame_idx / num_keyframes) * 2 - 1  # От -1 до 1
            height = height_factor * spiral_height_range / 2
            
            # Позиция камеры
            x = center[0] + radius * np.cos(angle)
            z = center[2] + radius * np.sin(angle)
            y = center[1] + height
            
            position = np.array([x, y, z])
            
            # Направление взгляда к центру
            direction = center - position
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0.01:
                direction = direction / direction_norm
                look_at = position + direction * 5.0  # Смотреть немного вперед
            else:
                look_at = position + np.array([0, 0, -1])
            
            keyframes.append({
                'position': position.tolist(),
                'lookAt': look_at.tolist(),
                'frame': frame_idx
            })
        
        print(f"Generated {len(keyframes)} keyframes for orbital path")
        
        return keyframes
    
    def _a_star_2d(self, walkable_positions: List[np.ndarray], 
                   start: np.ndarray, end: np.ndarray) -> List[np.ndarray]:
        """
        A* алгоритм на 2D XZ проекции
        
        Args:
            walkable_positions: Список проходимых позиций
            start: Начальная позиция
            end: Конечная позиция
            
        Returns:
            List[np.ndarray]: Путь из waypoints
        """
        if len(walkable_positions) < 2:
            return [start, end]
        
        # Проекция на XZ плоскость
        walkable_2d = np.array([[p[0], p[2]] for p in walkable_positions])
        start_2d = np.array([start[0], start[2]])
        end_2d = np.array([end[0], end[2]])
        
        # Поиск ближайших точек
        distances_start = np.linalg.norm(walkable_2d - start_2d, axis=1)
        distances_end = np.linalg.norm(walkable_2d - end_2d, axis=1)
        
        start_idx = np.argmin(distances_start)
        end_idx = np.argmin(distances_end)
        
        # Упрощенный A* (жадный поиск ближайших соседей)
        path_indices = [start_idx]
        current_idx = start_idx
        
        visited = set([start_idx])
        max_iterations = len(walkable_positions)
        iteration = 0
        
        while current_idx != end_idx and iteration < max_iterations:
            iteration += 1
            
            # Поиск ближайшего непосещенного соседа к цели
            unvisited = [i for i in range(len(walkable_2d)) if i not in visited]
            if not unvisited:
                break
            
            distances_to_end = np.linalg.norm(walkable_2d[unvisited] - end_2d, axis=1)
            next_idx = unvisited[np.argmin(distances_to_end)]
            
            path_indices.append(next_idx)
            visited.add(next_idx)
            current_idx = next_idx
        
        # Добавление конечной точки
        if end_idx not in path_indices:
            path_indices.append(end_idx)
        
        # Преобразование индексов в позиции
        waypoints = [walkable_positions[i] for i in path_indices]
        
        # Добавление начальной и конечной позиций
        if len(waypoints) > 0:
            waypoints[0] = start
            waypoints[-1] = end
        
        return waypoints
    
    def _interpolate_waypoints(self, waypoints: List[np.ndarray], 
                              num_keyframes: int) -> List[Dict]:
        """
        Интерполяция между waypoints с использованием Catmull-Rom сплайнов
        
        Args:
            waypoints: Список waypoints
            num_keyframes: Количество ключевых кадров
            
        Returns:
            List[Dict]: Интерполированные ключевые кадры
        """
        if len(waypoints) < 2:
            return []
        
        waypoints_array = np.array(waypoints)
        
        # Параметризация пути
        distances = np.zeros(len(waypoints))
        for i in range(1, len(waypoints)):
            distances[i] = distances[i-1] + np.linalg.norm(waypoints[i] - waypoints[i-1])
        
        if distances[-1] < 0.01:
            # Все точки совпадают
            return [{
                'position': waypoints[0].tolist(),
                'lookAt': (waypoints[0] + np.array([0, 0, -1])).tolist(),
                'frame': 0
            }]
        
        # Нормализация расстояний
        t = distances / distances[-1]
        
        keyframes = []
        
        for frame_idx in range(num_keyframes):
            frame_t = frame_idx / (num_keyframes - 1) if num_keyframes > 1 else 0.0
            
            # Интерполяция позиции
            position = self._catmull_rom_interpolate(waypoints_array, t, frame_t)
            
            # Вычисление направления взгляда (вперед по пути)
            if frame_idx < num_keyframes - 1:
                next_t = (frame_idx + 1) / (num_keyframes - 1) if num_keyframes > 1 else 1.0
                next_position = self._catmull_rom_interpolate(waypoints_array, t, next_t)
                direction = next_position - position
            else:
                direction = position - self._catmull_rom_interpolate(waypoints_array, t, max(0, frame_t - 0.01))
            
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0.01:
                direction = direction / direction_norm
                look_at = position + direction * 3.0  # Смотреть на 3 метра вперед
            else:
                look_at = position + np.array([0, 0, -1])
            
            keyframes.append({
                'position': position.tolist(),
                'lookAt': look_at.tolist(),
                'frame': frame_idx
            })
        
        return keyframes
    
    def _catmull_rom_interpolate(self, points: np.ndarray, t: np.ndarray, 
                                 target_t: float) -> np.ndarray:
        """
        Catmull-Rom интерполяция
        
        Args:
            points: Массив точек
            t: Параметры точек
            target_t: Целевой параметр для интерполяции
            
        Returns:
            np.ndarray: Интерполированная позиция
        """
        # Найти сегмент
        idx = np.searchsorted(t, target_t)
        idx = max(1, min(idx, len(points) - 2))
        
        # Catmull-Rom интерполяция между точками idx-1 и idx
        t0, t1 = t[idx-1], t[idx]
        if abs(t1 - t0) < 0.001:
            return points[idx]
        
        local_t = (target_t - t0) / (t1 - t0)
        local_t = np.clip(local_t, 0, 1)
        
        # Линейная интерполяция (упрощенная версия)
        return points[idx-1] * (1 - local_t) + points[idx] * local_t
    
    def _create_simple_path(self, num_keyframes: int) -> List[Dict]:
        """Создание простого пути при отсутствии проходимых ячеек"""
        min_bounds = np.min(self.points, axis=0)
        max_bounds = np.max(self.points, axis=0)
        center = (min_bounds + max_bounds) / 2.0
        
        keyframes = []
        for frame_idx in range(num_keyframes):
            t = frame_idx / (num_keyframes - 1) if num_keyframes > 1 else 0.0
            
            # Простой путь от одного угла к другому
            start = min_bounds + (max_bounds - min_bounds) * 0.2
            end = min_bounds + (max_bounds - min_bounds) * 0.8
            
            position = start * (1 - t) + end * t
            look_at = center
            
            keyframes.append({
                'position': position.tolist(),
                'lookAt': look_at.tolist(),
                'frame': frame_idx
            })
        
        return keyframes
    
    def _create_path_from_scene_points(self, num_keyframes: int, center: np.ndarray,
                                      min_bounds: np.ndarray, max_bounds: np.ndarray) -> List[Dict]:
        """Создание пути на основе точек сцены"""
        # Находим точки внутри центральной области (60% от bounds)
        inner_min = min_bounds + (max_bounds - min_bounds) * 0.2
        inner_max = min_bounds + (max_bounds - min_bounds) * 0.8
        
        # Фильтруем точки сцены
        mask = np.all((self.points >= inner_min) & (self.points <= inner_max), axis=1)
        inner_points = self.points[mask]
        
        if len(inner_points) == 0:
            inner_points = self.points
        
        # Выбираем несколько точек для пути
        num_waypoints = min(10, len(inner_points))
        indices = np.linspace(0, len(inner_points) - 1, num_waypoints, dtype=int)
        waypoints = [inner_points[i] for i in indices]
        
        # Интерполяция
        keyframes = self._interpolate_waypoints(waypoints, num_keyframes)
        return self._constrain_keyframes_to_scene(keyframes, min_bounds, max_bounds)
    
    def _constrain_keyframes_to_scene(self, keyframes: List[Dict], 
                                      min_bounds: np.ndarray, 
                                      max_bounds: np.ndarray) -> List[Dict]:
        """Ограничение ключевых кадров границами сцены"""
        constrained = []
        voxel_size = self.space_analyzer.voxel_size
        
        for kf in keyframes:
            pos = np.array(kf['position'])
            
            # Ограничиваем позицию bounds сцены
            pos = np.clip(pos, min_bounds - voxel_size, max_bounds + voxel_size)
            
            # Проверяем, что позиция близко к какой-то точке сцены
            distances, indices = self.kdtree.query(pos, k=5)
            if np.min(distances) > voxel_size * 3.0:
                # Если слишком далеко, перемещаем к ближайшей точке сцены
                nearest_point = self.points[indices[0]]
                pos = nearest_point + (pos - nearest_point) * 0.5  # На полпути
            
            # Обновляем lookAt, чтобы смотреть вперед по пути
            look_at = np.array(kf['lookAt'])
            direction = look_at - pos
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 0.1:
                direction = np.array([0, 0, -1])
            else:
                direction = direction / direction_norm
            
            # Ограничиваем lookAt тоже
            look_at = pos + direction * 3.0
            look_at = np.clip(look_at, min_bounds - voxel_size, max_bounds + voxel_size)
            
            constrained.append({
                'position': pos.tolist(),
                'lookAt': look_at.tolist(),
                'frame': kf['frame']
            })
        
        return constrained
    
    def save_keyframes(self, keyframes: List[Dict], output_path: str):
        """Сохранение ключевых кадров в JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({'keyframes': keyframes}, f, indent=2)
        
        print(f"Saved {len(keyframes)} keyframes to {output_path}")


if __name__ == "__main__":
    # Тестирование
    np.random.seed(42)
    points = np.random.rand(1000, 3) * 10
    
    # Создание фиктивного space_analyzer
    class FakeAnalyzer:
        def get_walkable_positions(self):
            return [np.array([1, 2, 3]), np.array([5, 2, 7]), np.array([9, 2, 11])]
    
    config = {
        'flythrough': {'waypoint_spacing': 2.0},
        'orbital': {'num_orbits': 3, 'spiral_height_range': 5.0}
    }
    
    planner = PathPlanner(points, [], FakeAnalyzer(), config)
    keyframes = planner.plan_orbital(np.array([5, 5, 5]), 10.0, 20)
    print(f"Generated {len(keyframes)} orbital keyframes")

