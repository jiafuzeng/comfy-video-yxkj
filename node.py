import os
import random
import time
import tempfile
import ffmpeg
import folder_paths
import av
import torch
import hashlib


def f32_pcm_yxkj(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to float 32 bits PCM format."""
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / (2 ** 15)
    elif wav.dtype == torch.int32:
        return wav.float() / (2 ** 31)
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")

def load_yxkj(filepath: str) -> tuple[torch.Tensor, int]:
    with av.open(filepath) as af:
        if not af.streams.audio:
            raise ValueError("No audio stream found in the file.")

        stream = af.streams.audio[0]
        sr = stream.codec_context.sample_rate
        n_channels = stream.channels

        frames = []
        length = 0
        for frame in af.decode(streams=stream.index):
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.shape[0] != n_channels:
                buf = buf.view(-1, n_channels).t()

            frames.append(buf)
            length += buf.shape[1]

        if not frames:
            raise ValueError("No audio frames decoded.")

        wav = torch.cat(frames, dim=1)
        wav = f32_pcm_yxkj(wav)
        return wav, sr


class VideoConcatNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_root": ("STRING", {"default": "", "multiline": False, "placeholder": "/path/to/videos"}),
                "label": ("STRING", {"default": "", "multiline": False, "placeholder": "标签/子目录名"}),
                "target_duration": ("FLOAT", {"default": 120.0, "min": 1.0, "max": 600.0, "step": 1.0, "display": "number", "tooltip": "拼接后总时长（秒）"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_video",)
    FUNCTION = "concat_videos"
    CATEGORY = "Video"

    def concat_videos(self, video_root, label, target_duration):
        # 获取目标目录
        target_dir = os.path.join(video_root, label)
        if not os.path.isdir(target_dir):
            raise Exception(f"目录不存在: {target_dir}")
        # 支持的视频格式
        video_exts = [".mp4", ".webm", ".mkv", ".mov", ".avi"]
        # 获取所有视频文件
        files = [f for f in os.listdir(target_dir) if os.path.splitext(f)[1].lower() in video_exts]
        if not files:
            raise Exception(f"子目录下没有可用视频文件: {target_dir}")
        # 打乱顺序
        random.shuffle(files)
        # 依次累加视频，直到总时长接近target_duration
        selected_paths = []
        total = 0.0
        for f in files:
            path = os.path.join(target_dir, f)
            try:
                probe = ffmpeg.probe(path)
                duration = float(probe['format']['duration'])
            except Exception:
                continue
            
            selected_paths.append(path)
            if total + duration > target_duration and selected_paths:
                break
            
        if not selected_paths:
            # 兜底：选第一个视频
            selected_paths = [os.path.join(target_dir, files[0])]
        # 在/tmp下创建临时目录
        temp_dir = tempfile.mkdtemp(dir='/tmp')
        # 生成唯一文件名
        timestamp = int(time.time() * 1000)
        random_num = random.randint(1, 9999)
        # 生成临时输出路径
        merged_path = os.path.join(temp_dir, f"concat_merged_{timestamp}_{random_num}.mp4")
        # 使用ffmpeg-python拼接视频（只拼接视频流，不处理音频）
        streams = []
        for path in selected_paths:
            streams.append(ffmpeg.input(path))
        concat_video = ffmpeg.concat(*[s.video for s in streams], v=1, a=0).node
        output = ffmpeg.output(concat_video[0], merged_path, an=None)
        ffmpeg.run(output, overwrite_output=True)
        # 获取合并后视频时长
        probe = ffmpeg.probe(merged_path)
        duration = float(probe['format']['duration'])
        output_path = merged_path
        # 若总时长超过target_duration，裁剪到target_duration
        if duration > target_duration:
            output_path = os.path.join(temp_dir, f"concat_result_{timestamp}_{random_num}.mp4")
            input_stream = ffmpeg.input(merged_path)
            output = ffmpeg.output(input_stream, output_path, t=target_duration, an=None)
            ffmpeg.run(output, overwrite_output=True)
            os.remove(merged_path)
        return (output_path,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 确保节点在输入变化时重新执行
        return float("nan")

class VideoAudioDurationNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("duration_sec",)
    FUNCTION = "get_duration"
    CATEGORY = "Video"

    def get_duration(self, audio):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        # waveform shape: [1, C, N] or [1, 1, N]
        num_samples = waveform.shape[-1]
        duration = float(num_samples) / float(sample_rate)
        return (duration,)



def strip_path(path):
    #This leaves whitespace inside quotes and only a single "
    #thus ' ""test"' -> '"test'
    #consider path.strip(string.whitespace+"\"")
    #or weightier re.fullmatch("[\\s\"]*(.+?)[\\s\"]*", path).group(1)
    path = path.strip()
    if path.startswith("\""):
        path = path[1:]
    if path.endswith("\""):
        path = path[:-1]
    return path

class LoadAudio_yxkj:
    @classmethod
    def INPUT_TYPES(s):
        #Hide ffmpeg formats if ffmpeg isn't available
        return {
            "required": {
                "audio_file": ("STRING", {"default": "input/", "vhs_path_extensions": ['wav','mp3','ogg','m4a','flac']}),
                },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = "audio_yxkj"
    FUNCTION = "load_audio"
    def load_audio(self, audio_file):
        audio_file = strip_path(audio_file)
        wav, sr = load_yxkj(audio_file)
        return ({'waveform': wav, 'sample_rate': sr},)

    @classmethod
    def IS_CHANGED(s):
        return True

    @classmethod
    def VALIDATE_INPUTS(s):
        return True