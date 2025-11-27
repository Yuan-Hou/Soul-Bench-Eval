# 字典数据类，存储视频文件路径和相应音频文件、文本文件、图片路径
import os
import json
from pathlib import Path

class VideoData:
    def __init__(self, video_path, audio_path = None, text_path = None, image_path = None):
        self.video_path = video_path.replace("\\", "/")
        self.audio_path = audio_path.replace("\\", "/") if audio_path else None
        self.text_path = text_path.replace("\\", "/") if text_path else None
        self.image_path = image_path.replace("\\", "/") if image_path else None

        self.text_data = {}
        if self.text_path.endswith('.json') and os.path.exists(self.text_path):
            with open(self.text_path, 'r', encoding='utf-8') as f:
                self.text_data = json.load(f)
        elif self.text_path and os.path.exists(self.text_path):
            with open(self.text_path, 'r', encoding='utf-8') as f:
                self.text_data = {"text": f.read()}
            

        self.video_filename = os.path.splitext(os.path.basename(video_path))[0]
        self.results = {}
    def has_audio(self):
        return self.audio_path is not None and os.path.exists(self.audio_path)
    def has_text(self):
        return self.text_path is not None and os.path.exists(self.text_path)
    def has_image(self):
        return self.image_path is not None and os.path.exists(self.image_path)
    def get_text(self):
        return self.text_data.get("en_prompt", self.text_data.get("text", "")) if self.text_data else ""
    def to_dict(self):
        return {
            "video_path": self.video_path,
            "audio_path": self.audio_path,
            "text_path": self.text_path,
            "text_data": self.text_data,
            "image_path": self.image_path,
            "results": self.results
        }
    def register_result(self, subject: str, result: dict):
        self.results[subject] = result