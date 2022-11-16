from youtube_dl import YoutubeDL
import pywhisper
from pywhisper.utils import write_srt, write_vtt, write_txt
import datetime
from pathlib import Path
import argparse
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator
import matchering as mg
from pydub import AudioSegment


def parse_args():
    parser = argparse.ArgumentParser()
    source_file = parser.add_mutually_exclusive_group(required=True)
    source_file.add_argument("--url", help="URL of the video to be transcribed")
    source_file.add_argument("--filename", help="Path to the video file to be transcribed")
    parser.add_argument("--model", type=str, default="base", help="Model to use")
    parser.add_argument("--lang", type=str, help="Language of the video")
    parser.add_argument("--audio_only", action="store_true", help="Video format to download")
    parser.add_argument("--output_format", type=str, default="srt", help="Subtitle format to output")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cpu or cuda)")
    parser.add_argument("--translate", action="store_true", help="Translate the video")
    parser.add_argument("--subtitles", action="store_true", help="Make subtitles")
    parser.add_argument("--prompt", type=str, default="", help="Initial prompt to help the model")
    parser.add_argument("--split", action="store_true", help="Use spleeter to separate audio")
    args = parser.parse_args()
    if args.device not in ["cpu", "cuda"]:
        raise ValueError("Invalid device")
    if args.model not in models:
        raise ValueError(f"Invalid model specified {args.model}, must be one of {models}")

    return args


models = ("tiny", "base", "small", "medium", "large")


def download_video(url, video_format):
    audio_dl = YoutubeDL({'format': video_format})
    filename = audio_dl.prepare_filename(audio_dl.extract_info(url))
    print(f"Video downloaded to {filename}")
    return filename


def spleeter_separate(filename):
    separator = Separator('spleeter:5stems')
    audio_loader = AudioAdapter.default()
    sr = 48000
    waveform, _ = audio_loader.load(filename, sample_rate=sr)
    separator.separate_to_file(audio_descriptor=filename, destination=Path(filename).parent,
                               filename_format="{filename}/{instrument}.{codec}",
                               codec="wav", offset=0.0, duration=None,
                               synchronous=True)
    print("Audio separated")


def speech_to_text(url=None, filename=None, lang=None, model="base", video_format="best", audio_only=False,
                   output_format="srt", device="cuda", translate=False, subtitles=False, prompt="", split=False):
    if url is None and filename is None:
        raise ValueError("Either url or filename must be specified")

    if url is not None:
        # download video
        filename = download_video(url, video_format if not audio_only else "bestaudio")

    if split:
        # separate audio
        spleeter_separate(filename)

    filename = make_subtitles(device, filename, lang, model, output_format, prompt, subtitles, translate)

    return filename

    # srt = create_srt(result)
    # write_subtitle_file(filename, srt)


def make_subtitles(device, filename, lang, model, output_format, prompt, subtitles, translate):
    if subtitles:
        # load whisper model and send it to GPU, transcribe audio to text
        model = pywhisper.load_model(model, device=device)
        result = model.transcribe(language=lang, audio=filename, verbose=True,
                                  condition_on_previous_text=True, initial_prompt=prompt,
                                  temperature=0.0, task="translate" if translate else "transcribe")
        filename = filename.replace(Path(filename).suffix, '.srt')
        create_srt_simple(filename, result["segments"], output_format)
        print(f"Subtitles written to {filename}")
    return filename


def create_srt_simple(filename, result, output_format):
    with open(filename, "w", encoding="utf-8") as f:
        write_srt(result, f)


def write_subtitle_file(filename, data):
    with open(f"{filename.replace(Path(filename).suffix, '.srt')}", "w", encoding="utf-8") as f:
        f.write(data)


def create_srt_data(result):
    srt = ""
    for i, s in enumerate(result["segments"]):
        seq = s["id"] + 1
        start = str.replace(str(datetime.timedelta(seconds=float(s["start"]))), ".", ",")
        end = str.replace(str(datetime.timedelta(seconds=float(s["end"]))), ".", ",")
        if "," not in start: start += ",000"
        if "," not in end: end += ",000"
        text = s["text"].strip().split(" ")
        # if len(text) > 4: // uncomment to break long lines
        #     text.insert(len(text) // 2 + 1, "{\a2}")
        text = " ".join(text).replace("\n ", "\n")
        srt += f"{seq}\n{start} --> {end}\n{text}\n\n"
    return srt


def main(**kwargs):
    speech_to_text(**kwargs)


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
    print("Done")
