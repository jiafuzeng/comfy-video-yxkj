from .node import VideoConcatNode,VideoAudioDurationNode,LoadAudio_yxkj

NODE_CLASS_MAPPINGS = {
    "VideoConcatNode": VideoConcatNode,
    "VideoAudioDurationNode": VideoAudioDurationNode,
    "LoadAudio_yxkj": LoadAudio_yxkj,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoConcatNode": "Video Concat (By Duration)",
    "VideoAudioDurationNode": "Video Audio Duration",
    "LoadAudio_yxkj": "Load Audio yxkj",
} 