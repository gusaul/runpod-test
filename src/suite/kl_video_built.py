import random
from typing import List
from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    CompositeVideoClip,
    TextClip,
    concatenate_videoclips,
    vfx
)
import pysrt

from kl_log import logger

def CreateVideoWithSubtitles(
    image_paths: List[str],
    audio_path: str,
    srt_path: str,
    output_path: str,
    video_fps: int,
    width: int,
    height: int
):
    """
    1) Determine random durations for each image (summing to total audio length).
    2) Apply random Ken Burns zoom for each.
    3) Concatenate to a single video, attach audio.
    4) Overlay subtitles from SRT.
    """
    logger.info("Creating final video with Ken Burns + subtitles...")

    # total audio length
    audio_clip = AudioFileClip(audio_path)
    total_audio_duration = audio_clip.duration

    # random durations for each image
    durations = random_durations(len(image_paths), total_audio_duration)

    # build image clips
    image_clips = []
    for img_path, dur in zip(image_paths, durations):
        clip = ImageClip(img_path, duration=dur).resize((width, height))

        # random ken burns
        zoom_start = random.uniform(1.0, 1.03)
        zoom_end = random.uniform(1.05, 1.15)

        def dynamic_zoom(t):
            return zoom_start + (zoom_end - zoom_start) * (t / dur)

        clip = clip.fx(vfx.resize, dynamic_zoom)
        image_clips.append(clip)

    # concatenate
    video_no_audio = concatenate_videoclips(image_clips, method="compose")
    # attach audio
    final_audio = audio_clip.set_duration(video_no_audio.duration)
    video_with_audio = video_no_audio.set_audio(final_audio)

    # add subtitles overlay
    final_video = overlay_subtitles(video_with_audio, srt_path)

    final_video.write_videofile(output_path, fps=video_fps, codec="libx264", audio_codec="aac")
    logger.info(f"Video written to: {output_path}")


def random_durations(n: int, total: float) -> List[float]:
    """
    Splits `total` seconds into n random durations that sum up to `total`.
    """
    if n <= 0:
        return []
    points = sorted([random.random() for _ in range(n - 1)])
    points = [0.0] + points + [1.0]
    durations = []
    for i in range(n):
        durations.append((points[i+1] - points[i]) * total)
    return durations


def overlay_subtitles(video_clip, srt_file: str):
    """
    Simple approach: parse SRT, create highlight text for start of each subtitle.
    """
    subs = pysrt.open(srt_file)
    text_clips = []

    for sub in subs:
        start_sec = (
            sub.start.hours * 3600 +
            sub.start.minutes * 60 +
            sub.start.seconds +
            sub.start.milliseconds / 1000
        )
        end_sec = (
            sub.end.hours * 3600 +
            sub.end.minutes * 60 +
            sub.end.seconds +
            sub.end.milliseconds / 1000
        )
        duration = end_sec - start_sec
        txt = sub.text.replace("\n", " ")

        # Normal text
        normal_clip = (TextClip(txt, fontsize=48, color='white', stroke_color='black', stroke_width=2)
                       .set_position(('center', 'bottom'))
                       .set_start(start_sec)
                       .set_duration(duration))

        # Highlight text (first 0.3s)
        highlight_clip = (TextClip(txt, fontsize=56, color='yellow', stroke_color='black', stroke_width=3)
                          .set_position(('center', 'bottom'))
                          .set_start(start_sec)
                          .set_duration(min(0.3, duration)))

        text_clips.extend([highlight_clip, normal_clip])

    return CompositeVideoClip([video_clip, *text_clips])
