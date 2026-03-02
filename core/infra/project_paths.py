"""项目路径管理器 - 集中管理所有项目相关的文件路径。"""

import os
import shutil
from typing import Optional


class ProjectPaths:
    """项目路径管理器，统一管理项目文件结构。"""

    _OPENING_EXTENSIONS = [
        ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv", ".png", ".jpg", ".jpeg"
    ]

    def __init__(self, project_dir: str):
        self.root = project_dir
        self.text = os.path.join(project_dir, "text")
        self.media = os.path.join(project_dir, "media")
        self.voice = os.path.join(project_dir, "voice")
        self._legacy_images_dir = os.path.join(project_dir, "images")

    # ==================== Text 目录相关路径 ====================

    def raw_json(self) -> str:
        return os.path.join(self.text, "raw.json")

    def raw_docx(self) -> str:
        return os.path.join(self.text, "raw.docx")

    def script_json(self) -> str:
        return os.path.join(self.text, "script.json")

    def script_docx(self) -> str:
        return os.path.join(self.text, "script.docx")

    def keywords_json(self) -> str:
        return os.path.join(self.text, "keywords.json")

    def mini_summary_json(self) -> str:
        return os.path.join(self.text, "mini_summary.json")

    def media_plan_json(self) -> str:
        return os.path.join(self.text, "media_plan.json")

    # ==================== Media 目录相关路径 ====================

    @property
    def images(self) -> str:
        """兼容旧调用：images 目录别名，实际映射到 media。"""
        return self.media

    def opening_image(self) -> str:
        """开场素材路径（自动检测图片或视频格式）。"""
        for ext in self._OPENING_EXTENSIONS:
            path = os.path.join(self.media, f"opening{ext}")
            if os.path.exists(path):
                return path
        return os.path.join(self.media, "opening.png")

    def segment_image(self, index: int) -> str:
        return os.path.join(self.media, f"segment_{index}.png")

    def cover_image(self, suffix: str) -> str:
        return os.path.join(self.root, f"cover_{suffix}.png")

    # ==================== Voice 目录相关路径 ====================

    def opening_audio(self) -> str:
        return os.path.join(self.voice, "opening.mp3")

    def segment_audio(self, index: int, extension: str = "mp3") -> str:
        return os.path.join(self.voice, f"voice_{index}.{extension}")

    def srt_subtitles(self) -> str:
        return os.path.join(self.voice, "字幕.srt")

    # ==================== 其他文件路径 ====================

    def final_video(self) -> str:
        return os.path.join(self.root, "final_video.mp4")

    # ==================== 工具方法 ====================

    def _resolve_non_conflict_path(self, target_path: str) -> str:
        if not os.path.exists(target_path):
            return target_path

        dirname, filename = os.path.split(target_path)
        stem, ext = os.path.splitext(filename)
        index = 1
        while True:
            candidate = os.path.join(dirname, f"{stem}_migrated{index}{ext}")
            if not os.path.exists(candidate):
                return candidate
            index += 1

    def _migrate_images_to_media(self) -> None:
        legacy = self._legacy_images_dir
        if not os.path.isdir(legacy):
            return

        if not os.path.exists(self.media):
            os.rename(legacy, self.media)
            return

        for name in os.listdir(legacy):
            source = os.path.join(legacy, name)
            target = os.path.join(self.media, name)
            target = self._resolve_non_conflict_path(target)
            shutil.move(source, target)

        try:
            os.rmdir(legacy)
        except OSError:
            pass

    def ensure_dirs_exist(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.text, exist_ok=True)
        os.makedirs(self.voice, exist_ok=True)
        self._migrate_images_to_media()
        os.makedirs(self.media, exist_ok=True)

    def segment_image_exists(self, index: int) -> bool:
        return os.path.exists(self.segment_image(index))

    def segment_audio_exists(self, index: int) -> Optional[str]:
        wav_path = self.segment_audio(index, "wav")
        if os.path.exists(wav_path):
            return wav_path

        mp3_path = self.segment_audio(index, "mp3")
        if os.path.exists(mp3_path):
            return mp3_path

        return None


__all__ = ["ProjectPaths"]
