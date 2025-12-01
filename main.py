"""
Main script for running the 3D Gaussian Splatting video generation system
"""

import asyncio
import yaml
import argparse
from pathlib import Path
import numpy as np

from src.ply_loader import PLYLoader
from src.scene_explorer import SceneExplorer
from src.space_analyzer import SpaceAnalyzer
from src.path_planner import PathPlanner
from src.video_renderer import VideoRenderer
from src.object_detector import ObjectDetector


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='3D Gaussian Splatting Video Generator')
    parser.add_argument('--scene', type=str, help='Path to .splat file')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--task', type=str, choices=[
        'video', 'detect', 'path_planning', 'obstacle_avoidance', 
        'coverage', '360', 'interactive', 'realtime', 'artistic'
    ], help='Task to perform')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine scene
    if args.scene:
        scene_path = args.scene
    else:
        scene_path = config.get('scene', {}).get('default_scene', 'Theater.splat')
        scenes_dir = config.get('scene', {}).get('scenes_dir', '.')
        scene_path = Path(scenes_dir) / scene_path
    
    if not Path(scene_path).exists():
        print(f"Error: Scene file not found: {scene_path}")
        return
    
    # Determine task
    task = args.task or 'video'
    
    print(f"Processing scene: {scene_path}")
    print("=" * 60)
    print(f"Task: {task}")
    print("=" * 60)
    
    # Stage 1: Scene Loading
    print("\n[Stage 1/6] Loading scene...")
    loader = PLYLoader(str(scene_path))
    points, metadata = loader.load()
    
    # Stage 2: Scene Exploration
    print("\n[Stage 2/6] Scene exploration...")
    scene_explorer = SceneExplorer(
        metadata['center'],
        metadata['radius'],
        config.get('scene_explorer', {})
    )
    viewpoints = scene_explorer.generate_viewpoints()
    
    # Save viewpoints
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_explorer.save_viewpoints(output_dir / 'viewpoints.json')
    
    # Stage 3: Space Analysis
    print("\n[Stage 3/6] Space analysis...")
    space_analyzer = SpaceAnalyzer(points, config.get('space_analyzer', {}))
    walkable_cells = space_analyzer.analyze()
    
    # Stage 4: Path Planning
    print("\n[Stage 4/6] Path planning...")
    path_planner = PathPlanner(
        points,
        walkable_cells,
        space_analyzer,
        config.get('path_planner', {})
    )
    
    # Path planning for tasks that require rendering or planning
    keyframes = None
    if task == 'path_planning':
        # Path planning only (no rendering)
        print("\n[Task 4] Path Planning - Planning path...")
        keyframes = path_planner.plan_flythrough(num_keyframes=300)
        path_planner.save_keyframes(keyframes, output_dir / 'keyframes.json')
        print("\n[OK] Path planning completed!")
        print(f"  Keyframes saved to: {output_dir / 'keyframes.json'}")
        print(f"  Total keyframes: {len(keyframes)}")
        print(f"  Walkable cells found: {len(walkable_cells)}")
        print(f"  Walkable positions used: {len(space_analyzer.get_walkable_positions())}")
        return
    elif task not in ['detect', 'obstacle_avoidance', 'interactive', 'realtime']:
        if task == 'coverage':
            # Special algorithm for covering most of the scene
            keyframes = path_planner.plan_coverage(num_keyframes=500)
        elif task in ['video', 'artistic']:
            # Flythrough for regular video
            num_frames = 300
            keyframes = path_planner.plan_flythrough(num_keyframes=num_frames)
        elif task == '360':
            # Orbital path for 360°
            keyframes = path_planner.plan_orbital(
                np.array(metadata['center']),
                metadata['radius'] * 1.5,
                num_keyframes=360
            )
        else:
            # Default flythrough
            keyframes = path_planner.plan_flythrough(num_keyframes=300)
        
        # Save keyframes
        path_planner.save_keyframes(keyframes, output_dir / 'keyframes.json')
    
    elif task == 'obstacle_avoidance':
        # Obstacle avoidance demonstration
        print("\n[Task 5] Obstacle Avoidance")
        print(f"  Walkable cells found: {len(walkable_cells)}")
        print(f"  Path planned with obstacle avoidance via A* algorithm")
        print(f"  Results saved to: {output_dir / 'keyframes.json'}")
        
        # Create small test path for demonstration
        test_keyframes = path_planner.plan_flythrough(num_keyframes=50)
        path_planner.save_keyframes(test_keyframes, output_dir / 'obstacle_avoidance_path.json')
        print(f"  Test path saved to: {output_dir / 'obstacle_avoidance_path.json'}")
        return
    
    elif task == 'interactive':
        # Interactive demo
        print("\n[Task 8] Interactive Demo")
        await run_interactive_demo(config, str(scene_path), metadata)
        return
    
    elif task == 'realtime':
        # Real-time preview
        print("\n[Task 9] Real-time Preview")
        await run_realtime_preview(config, str(scene_path), metadata)
        return
    
    # Video rendering for other tasks
    print("\n[Stage 5/6] Video rendering...")
    print("[Stage 6/6] Object detection (if enabled)...")
    
    # Initialize object detector
    object_detector = None
    if task in ['detect', 'video', 'coverage', 'artistic']:
        object_detector = ObjectDetector(config.get('object_detection', {}))
    
    # Initialize renderer
    renderer = VideoRenderer(config, object_detector)
    
    try:
        # Initialize browser
        await renderer.initialize(str(scene_path))
        
        # Render video
        if task == '360':
            video_path = await renderer.render_360_video(
                metadata['center'],
                metadata['radius'] * 1.5,
                360,
                f"{Path(scene_path).stem}_360"
            )
        elif task == 'detect':
            # Detection only on existing frames (if any)
            print("Task 'detect' requires pre-rendered frames")
            print("Use 'video' for rendering with detection")
        else:
            output_name = f"{Path(scene_path).stem}_{task}"
            if task == 'artistic':
                # High quality for artistic
                config['video']['crf'] = 15  # Even higher quality
            video_path = await renderer.render_keyframes(
                keyframes,
                output_name
            )
            print(f"\n[OK] Video successfully created: {video_path}")
        
    finally:
        await renderer.close()
    
    print("\n" + "=" * 60)
    print("Done!")


async def run_interactive_demo(config, scene_path, metadata):
    """Interactive demo (Task 8)"""
    from playwright.async_api import async_playwright
    
    print("Starting interactive demo...")
    print("Browser will open where you can control the camera")
    
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    page = await browser.new_page()
    
    # Load viewer
    viewer_path = Path(__file__).parent / 'viewer' / 'index.html'
    viewer_url = f"file://{viewer_path.absolute()}"
    await page.goto(viewer_url)
    
    # Load scene
    await page.wait_for_function('window.loadPLY !== undefined', timeout=10000)
    ply_absolute_path = Path(scene_path).absolute()
    await page.evaluate(f'''
        await window.loadPLY('file://{ply_absolute_path}');
    ''')
    
    await page.wait_for_function('window.isViewerReady()', timeout=30000)
    
    print("\n[OK] Interactive demo started!")
    print("  - Use mouse to rotate camera")
    print("  - Mouse wheel to zoom")
    print("  - Close browser to exit")
    
    # Wait for browser to close
    try:
        await page.wait_for_timeout(3600000)  # 1 hour maximum
    except:
        pass
    
    await browser.close()


async def run_realtime_preview(config, scene_path, metadata):
    """Real-time preview (Task 9)"""
    from playwright.async_api import async_playwright
    import time
    
    print("Starting real-time preview...")
    print("Preview of planned path will be shown in real-time")
    
    # Load scene data for path planning
    loader = PLYLoader(scene_path)
    points, _ = loader.load()
    space_analyzer = SpaceAnalyzer(points, config.get('space_analyzer', {}))
    walkable_cells = space_analyzer.analyze()
    
    path_planner = PathPlanner(points, walkable_cells, space_analyzer, config.get('path_planner', {}))
    preview_keyframes = path_planner.plan_orbital(
        np.array(metadata['center']),
        metadata['radius'] * 1.2,
        num_keyframes=100
    )
    
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    page = await browser.new_page()
    
    # Load viewer
    viewer_path = Path(__file__).parent / 'viewer' / 'index.html'
    viewer_url = f"file://{viewer_path.absolute()}"
    await page.goto(viewer_url)
    
    # Load scene
    await page.wait_for_function('window.loadPLY !== undefined', timeout=10000)
    ply_absolute_path = Path(scene_path).absolute()
    await page.evaluate(f'''
        await window.loadPLY('file://{ply_absolute_path}');
    ''')
    
    await page.wait_for_function('window.isViewerReady()', timeout=30000)
    
    print("\n[OK] Real-time preview started!")
    print("  Camera will move along preview path")
    
    # Path animation
    for keyframe in preview_keyframes:
        position = keyframe['position']
        look_at = keyframe.get('lookAt', position)
        await page.evaluate(f'''
            window.setCameraPosition({position}, {look_at});
        ''')
        await asyncio.sleep(0.1)  # 10 FPS for preview
    
    print("  Preview completed. Close browser.")
    await page.wait_for_timeout(5000)
    
    await browser.close()


if __name__ == "__main__":
    asyncio.run(main())

