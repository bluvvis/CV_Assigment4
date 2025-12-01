"""
Video Renderer - Захват кадров через браузер и сборка видео
Этап 5: Рендеринг видео
"""

import asyncio
import json
import subprocess
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from playwright.async_api import async_playwright, Page, Browser
import time
from tqdm import tqdm

from src.object_detector import ObjectDetector


class VideoRenderer:
    """Рендеринг видео через Playwright и FFmpeg"""
    
    def __init__(self, config: Dict, object_detector: Optional[ObjectDetector] = None):
        """
        Инициализация рендерера видео
        
        Args:
            config: Конфигурация из config.yaml
            object_detector: Опциональный детектор объектов
        """
        self.config = config
        self.rendering_config = config.get('rendering', {})
        self.video_config = config.get('video', {})
        self.object_detector = object_detector
        
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.playwright = None  # Ссылка на playwright для правильного закрытия
        
        # Создание директорий для вывода
        self.frames_dir = Path(self.rendering_config.get('output_frames_dir', 'output/frames'))
        self.videos_dir = Path(self.rendering_config.get('output_videos_dir', 'output/videos'))
        self.reports_dir = Path(self.rendering_config.get('output_reports_dir', 'output/reports'))
        
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self, ply_path: str):
        """
        Инициализация браузера и загрузка сцены
        
        Args:
            ply_path: Путь к PLY файлу
        """
        print("Initializing browser...")
        
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.rendering_config.get('browser_headless', True)
        )
        
        # Создание страницы
        self.page = await self.browser.new_page()
        
        # Установка размера окна
        width = self.video_config.get('width', 1920)
        height = self.video_config.get('height', 1080)
        await self.page.set_viewport_size({'width': width, 'height': height})
        
        # Убеждаемся, что размер установлен правильно
        viewport = self.page.viewport_size
        if viewport and (viewport.get('width') != width or viewport.get('height') != height):
            print(f"Warning: Viewport size is {viewport}, expected {width}x{height}")
        
        # Загрузка viewer
        viewer_path = Path(__file__).parent.parent / 'viewer' / 'index.html'
        viewer_url = f"file://{viewer_path.absolute()}"
        
        print(f"Loading viewer from {viewer_url}")
        await self.page.goto(viewer_url)
        
        # Ожидание загрузки Three.js и viewer
        print("Waiting for Three.js to load...")
        await asyncio.sleep(2)  # Даем время на загрузку скриптов
        
        # Проверяем что Three.js загружен
        three_loaded = await self.page.evaluate('typeof THREE !== "undefined"')
        if not three_loaded:
            raise RuntimeError("Three.js failed to load")
        
        print("Three.js loaded successfully")
        
        # Загрузка точек сцены
        print(f"Loading scene data...")
        
        # Загружаем точки через Python и передаем в браузер
        from src.ply_loader import PLYLoader
        loader = PLYLoader(ply_path)
        points, metadata = loader.load()
        
        # Берем выборку точек для рендеринга (слишком много для браузера)
        sample_rate = max(1, len(points) // 300_000)  # Максимум 300k точек
        sampled_points = points[::sample_rate]
        
        print(f"  Loaded {len(points):,} points, using {len(sampled_points):,} for rendering")
        
        # Передаем точки в браузер
        points_list = sampled_points.tolist()
        center = metadata['center']
        radius = metadata['radius']
        
        print("  Uploading points to browser...")
        await self.page.evaluate(f'''
            window.loadPoints({points_list}, {center}, {radius});
        ''')
        
        # Ожидание готовности
        print("  Waiting for scene to render...")
        await self.page.wait_for_function('window.isViewerReady()', timeout=30000)
        await asyncio.sleep(2)  # Дополнительная пауза для рендеринга
        print("  Scene ready!")
        
        print("Viewer initialized and PLY loaded")
    
    async def render_keyframes(self, keyframes: List[Dict], output_name: str) -> str:
        """
        Рендеринг ключевых кадров
        
        Args:
            keyframes: Список ключевых кадров
            output_name: Имя выходного файла (без расширения)
            
        Returns:
            str: Путь к созданному видео файлу
        """
        print(f"Rendering {len(keyframes)} keyframes...")
        print(f"Output: {self.videos_dir / f'{output_name}.mp4'}")
        print(f"Progress will be shown below:\n")
        
        frame_paths = []
        all_detections = []
        
        screenshot_delay = self.rendering_config.get('screenshot_delay', 0.5)
        
        import time
        start_time = time.time()
        
        for idx, keyframe in enumerate(tqdm(keyframes, desc="Rendering frames", unit="frame")):
            # Установка позиции камеры
            position = keyframe['position']
            look_at = keyframe.get('lookAt', [position[0], position[1], position[2] - 1])
            
            # Правильная сериализация в JSON для JavaScript
            import json
            position_json = json.dumps(position)
            look_at_json = json.dumps(look_at)
            
            await self.page.evaluate(f'''
                window.setCameraPosition({position_json}, {look_at_json});
            ''')
            
            # Ожидание рендеринга
            await asyncio.sleep(screenshot_delay)
            
            # Скриншот
            frame_path = self.frames_dir / f"{output_name}_frame_{idx:05d}.png"
            await self.page.screenshot(path=str(frame_path))
            frame_paths.append(frame_path)
            
            # Детекция объектов (если включена)
            if self.object_detector:
                detections, vis_image = self.object_detector.detect(str(frame_path))
                if detections:
                    all_detections.append({
                        'frame': idx,
                        'detections': detections
                    })
                
                # Сохранение визуализированного изображения
                if vis_image:
                    vis_image.save(str(frame_path))
            
            # Периодический вывод статуса
            if (idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                fps_actual = (idx + 1) / elapsed
                remaining = (len(keyframes) - idx - 1) / fps_actual if fps_actual > 0 else 0
                print(f"  [{idx+1}/{len(keyframes)}] FPS: {fps_actual:.2f} | "
                      f"Elapsed: {elapsed/60:.1f}min | Remaining: ~{remaining/60:.1f}min")
        
        elapsed_total = time.time() - start_time
        print(f"\n✓ All {len(keyframes)} frames rendered in {elapsed_total/60:.1f} minutes")
        
        # Сохранение отчета о детекциях
        if all_detections:
            report_path = self.reports_dir / f"{output_name}_detections.json"
            with open(report_path, 'w') as f:
                json.dump({'detections': all_detections}, f, indent=2)
            print(f"Saved detection report to {report_path}")
        
        # Кодирование видео
        video_path = await self._encode_video(frame_paths, output_name)
        
        print(f"Video saved to {video_path}")
        
        return str(video_path)
    
    async def _encode_video(self, frame_paths: List[Path], output_name: str) -> Path:
        """
        Кодирование кадров в видео через FFmpeg
        
        Args:
            frame_paths: Список путей к кадрам
            output_name: Имя выходного файла
            
        Returns:
            Path: Путь к созданному видео файлу
        """
        print(f"\n[Этап кодирования] Encoding {len(frame_paths)} frames with FFmpeg...")
        import time
        encode_start = time.time()
        
        video_path = self.videos_dir / f"{output_name}.mp4"
        
        # Параметры FFmpeg
        fps = self.video_config.get('fps', 30)
        codec = self.video_config.get('codec', 'libx264')
        crf = self.video_config.get('crf', 18)
        
        # Создание списка файлов для FFmpeg
        # FFmpeg требует файлы в формате pattern или список файлов
        first_frame = frame_paths[0]
        pattern = str(first_frame).replace('_frame_00000.png', '_frame_%05d.png')
        
        # Поиск FFmpeg (может быть в PATH или в Playwright)
        ffmpeg_path = None
        
        # Попытка найти ffmpeg в PATH
        import shutil
        ffmpeg_path = shutil.which('ffmpeg')
        
        # Если не найден, пробуем встроенный из Playwright
        if not ffmpeg_path:
            import os
            local_appdata = os.getenv('LOCALAPPDATA', '')
            if local_appdata:
                playwright_ffmpeg = Path(local_appdata) / 'ms-playwright' / 'ffmpeg-1011' / 'ffmpeg.exe'
                if playwright_ffmpeg.exists():
                    ffmpeg_path = str(playwright_ffmpeg)
                    print(f"Using Playwright's FFmpeg: {ffmpeg_path}")
            
            # Пробуем другие возможные пути
            if not ffmpeg_path:
                for possible_path in [
                    Path.home() / '.cache' / 'ms-playwright' / 'ffmpeg-1011' / 'ffmpeg.exe',
                    Path.home() / 'AppData' / 'Local' / 'ms-playwright' / 'ffmpeg-1011' / 'ffmpeg.exe',
                    Path('C:/ffmpeg/bin/ffmpeg.exe'),
                ]:
                    if possible_path.exists():
                        ffmpeg_path = str(possible_path)
                        print(f"Found FFmpeg at: {ffmpeg_path}")
                        break
        
        # Если FFmpeg не найден, используем OpenCV как fallback
        if not ffmpeg_path:
            print("FFmpeg not found, using OpenCV for video encoding (slower but works)...")
            return await self._encode_video_opencv(frame_paths, output_name)
        
        # Команда FFmpeg
        cmd = [
            ffmpeg_path,
            '-y',  # Перезаписать выходной файл
            '-framerate', str(fps),
            '-i', pattern,
            '-c:v', codec,
            '-crf', str(crf),
            '-pix_fmt', 'yuv420p',
            '-r', str(fps),  # Output frame rate
            str(video_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            encode_time = time.time() - encode_start
            print(f"[OK] FFmpeg encoding completed in {encode_time:.1f} seconds")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] FFmpeg error: {e.stderr}")
            raise
        except FileNotFoundError:
            # Если FFmpeg не работает, используем OpenCV
            print("FFmpeg failed, trying OpenCV fallback...")
            return await self._encode_video_opencv(frame_paths, output_name)
        
        return video_path
    
    async def _encode_video_opencv(self, frame_paths: List[Path], output_name: str) -> Path:
        """Кодирование видео через OpenCV (fallback если FFmpeg недоступен)"""
        import cv2
        
        print("Encoding video with OpenCV...")
        encode_start = time.time()
        
        video_path = self.videos_dir / f"{output_name}.mp4"
        fps = self.video_config.get('fps', 30)
        width = self.video_config.get('width', 1920)
        height = self.video_config.get('height', 1080)
        
        # Читаем первый кадр для определения размера
        first_frame = cv2.imread(str(frame_paths[0]))
        if first_frame is None:
            raise RuntimeError(f"Failed to read first frame: {frame_paths[0]}")
        
        actual_height, actual_width = first_frame.shape[:2]
        
        # Используем фактический размер кадра
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (actual_width, actual_height))
        
        if not out.isOpened():
            raise RuntimeError(f"Failed to open video writer for {video_path}")
        
        print(f"Writing {len(frame_paths)} frames at {fps} fps...")
        
        for frame_path in tqdm(frame_paths, desc="Encoding frames"):
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                # Изменяем размер если нужно
                if frame.shape[:2] != (actual_height, actual_width):
                    frame = cv2.resize(frame, (actual_width, actual_height))
                out.write(frame)
        
        out.release()
        encode_time = time.time() - encode_start
        print(f"[OK] OpenCV encoding completed in {encode_time:.1f} seconds")
        
        return video_path
    
    async def render_360_video(self, center: List[float], radius: float, 
                               num_frames: int, output_name: str) -> str:
        """
        Рендеринг 360° видео
        
        Args:
            center: Центр сцены
            radius: Радиус орбиты
            num_frames: Количество кадров
            output_name: Имя выходного файла
            
        Returns:
            str: Путь к созданному видео файлу
        """
        print(f"Rendering 360° video with {num_frames} frames...")
        
        keyframes = []
        for frame_idx in range(num_frames):
            angle = (frame_idx / num_frames) * 2 * np.pi
            
            x = center[0] + radius * np.cos(angle)
            z = center[2] + radius * np.sin(angle)
            y = center[1]
            
            position = [x, y, z]
            look_at = center
            
            keyframes.append({
                'position': position,
                'lookAt': look_at,
                'frame': frame_idx
            })
        
        return await self.render_keyframes(keyframes, output_name)
    
    async def close(self):
        """Закрытие браузера и playwright"""
        try:
            if self.browser:
                await self.browser.close()
                self.browser = None
            if self.playwright:
                await self.playwright.stop()
                self.playwright = None
            print("Browser closed")
        except Exception as e:
            # Игнорируем ошибки при закрытии (ресурсы уже могут быть закрыты)
            pass


if __name__ == "__main__":
    # Тестирование (требует async контекст)
    import sys
    
    async def test():
        config = {
            'video': {'width': 1920, 'height': 1080, 'fps': 30},
            'rendering': {'browser_headless': True, 'screenshot_delay': 0.5}
        }
        
        renderer = VideoRenderer(config)
        
        if len(sys.argv) > 1:
            await renderer.initialize(sys.argv[1])
            
            # Test keyframe
            keyframes = [{
                'position': [0, 0, 10],
                'lookAt': [0, 0, 0],
                'frame': 0
            }]
            
            await renderer.render_keyframes(keyframes, 'test')
            await renderer.close()
        else:
            print("Usage: python video_renderer.py <ply_path>")
    
    asyncio.run(test())

