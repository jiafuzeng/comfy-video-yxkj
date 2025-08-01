from .node import VideoConcatNode,VideoAudioDurationNode

NODE_CLASS_MAPPINGS = {
    "VideoConcatNode": VideoConcatNode,
    "VideoAudioDurationNode": VideoAudioDurationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoConcatNode": "Video Concat (By Duration)",
    "VideoAudioDurationNode": "Video Audio Duration",
} 