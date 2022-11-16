from youtube_dl import YoutubeDL
import pywhisper
from pywhisper.utils import write_srt, write_vtt, write_txt
import datetime
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # only one of the following two arguments is allowed at a time
    source_file = parser.add_mutually_exclusive_group(required=True)
    source_file.add_argument("-u", "--url", help="URL of the video to be transcribed")
    source_file.add_argument("-f", "--filename", help="Path to the video file to be transcribed")
    parser.add_argument("--model", type=str, default="BASE", help="Model to use")
    parser.add_argument("--lang", type=str, default="en", help="Language of the video")
    parser.add_argument("--video_format", type=str, default="best", help="Video format to download")
    parser.add_argument("--output_format", type=str, default="srt", help="Subtitle format to output")
    return parser.parse_args()


models = ("tiny", "base", "small", "medium", "large")


def speech_to_text(url=None, filename=None, model="base", lang="en", video_format="best", output_format="srt"):
    if url is None and filename is None:
        raise ValueError("Either url or filename must be specified")

    if model not in models:
        raise ValueError(f"Invalid model specified {model}")

    # download video
    if url is not None:
        audio_dl = YoutubeDL({'format': video_format})
        # audio_dl.extract_info(url)
        filename = audio_dl.prepare_filename(audio_dl.extract_info(url))

    # load whisper model and send it to GPU, transcribe audio to text
    model = pywhisper.load_model(model, device="cuda")
    result = model.transcribe(language=lang, audio=filename, verbose=True,
                              condition_on_previous_text=True, initial_prompt="pohjavesi",
                              temperature=0.0)
    filename = filename.replace(Path(filename).suffix, '.srt')
    create_srt_simple(filename, result["segments"], output_format)
    return filename
    # write srt file
    # exit(0)

    # srt = create_srt(result)
    # write_subtitle_file(filename, srt)


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
