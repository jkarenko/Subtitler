import math

import numpy as np
from yt_dlp import YoutubeDL
import pywhisper
from pywhisper.utils import write_srt, write_vtt, write_txt
import datetime
from pathlib import Path
import argparse
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator
import keyring
# import os
import openai
from summarizer import Summarizer, TransformerSummarizer
import time
import noisereduce as nr
from scipy.io import wavfile

# import matchering as mg
# from pydub import AudioSegment


models = ("tiny", "base", "small", "medium", "large")
# openai.organization = os.getenv("OPENAI_ORG")
# openai.api_key = os.getenv("OPENAI_API_KEY")
# get openai_api_token from the keychain
openai.api_key = keyring.get_keyring().get_password("openai-api-token", "openai")


def parse_args():
    parser = argparse.ArgumentParser()
    source_file = parser.add_mutually_exclusive_group(required=True)
    source_file.add_argument("--url", help="URL of the video to be transcribed")
    source_file.add_argument("--filename", help="Path to the video file to be transcribed")
    source_file.add_argument("--text_file", help="Text to be summarised")
    source_file.add_argument("--denoise", nargs="+", help="Denoise audio files")
    parser.add_argument("--mix", type=str, help="Output file name of the mixed audio")
    parser.add_argument("--model", type=str, default="base", help="Model to use")
    parser.add_argument("--lang", type=str, help="Language of the video")
    parser.add_argument("--audio_only", action="store_true", help="Video format to download")
    parser.add_argument("--output_format", type=str, default="srt", help="Subtitle format to output")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--translate", action="store_true", help="Translate the video")
    parser.add_argument("--subtitles", action="store_true", help="Make subtitles")
    parser.add_argument("--prompt", type=str, default="", help="Initial prompt to help the model")
    parser.add_argument("--split", action="store_true", help="Use spleeter to separate audio")
    parser.add_argument("--summary", action="store_true", help="Summarise the inferred text")
    summariser_model = parser.add_mutually_exclusive_group()
    summariser_model.add_argument("--bert", action="store_true", help="Use BERT to summarise the inferred text")
    summariser_model.add_argument("--gpt2", action="store_true", help="Use GPT2 to summarise the inferred text")
    summariser_model.add_argument("--xlnet", action="store_true", help="Use XLNet to summarise the inferred text")
    args = parser.parse_args()

    if args.device not in ["cpu", "cuda"]:
        raise ValueError("Invalid device")
    if args.model not in models:
        raise ValueError(f"Invalid model specified {args.model}, must be one of {models}")

    return args


def summarise_text(text,
                   max_tokens=2048,
                   temperature=0.1,
                   top_p=1.0,
                   frequency_penalty=0.0,
                   presence_penalty=0.0,
                   stop="###"):
    if args.bert:
        model = Summarizer()
        return model(text, min_length=30, max_length=100)
    if args.gpt2:
        model = TransformerSummarizer(transformer_type="GPT2", transformer_model_key="gpt2-medium")
        return "".join(model(text, min_length=30, max_length=100))
    if args.xlnet:
        model = TransformerSummarizer(transformer_type="XLNet", transformer_model_key="xlnet-base-cased")
        return "".join(model(text, min_length=30, max_length=100))
    # count words in text, 1 token ~= 0.75 words, split into parts if tokens + max_tokens >= 4096
    words_to_tokens_conversion_rate = .75
    words_per_part = math.floor(max_tokens * words_to_tokens_conversion_rate)
    words = text.split()
    num_parts = math.ceil(len(words) / words_per_part)
    summaries = []
    if num_parts > 1:
        print(f"Text too long, splitting into {num_parts} parts")
        text_parts = [" ".join(words[i * words_per_part:(i + 1) * words_per_part]) for i in range(num_parts)]
        summaries = [summarise_text(part) for part in text_parts]
    if summaries:
        text = "\n".join(summaries)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        # prompt=f"{text}\nTo re-iterate, these are the steps:{stop}",
        # prompt=f"{text}\nIn summary:{stop}",
        messages=[
            {"role": "system", "content": "You are a text summarizer. Summarize the following text:"},
            {"role": "user", "content": text},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop
    )
    return response.choices[0].message.content


def download_video(url, video_format):
    with YoutubeDL({'format': video_format}) as ydl:
        # info = ydl.extract_info(url, download=False)
        filename = ydl.prepare_filename(ydl.extract_info(url))
    print(f"Video downloaded to {filename}")
    return filename


def spleeter_separate(filename):
    separator = Separator('spleeter:2stems')
    audio_loader = AudioAdapter.default()
    sr = 48000
    waveform, _ = audio_loader.load(filename, sample_rate=sr)
    separator.separate_to_file(audio_descriptor=filename, destination=Path(filename).parent,
                               filename_format="{filename}/{instrument}.{codec}",
                               codec="wav", offset=0.0, duration=None,
                               synchronous=True)
    print("Audio separated")


def speech_to_text(url=None, filename=None, denoise=None, mix=None, lang=None, model="base", video_format="best",
                   audio_only=False,
                   output_format="srt", device="cuda", translate=False, subtitles=False, prompt="", split=False,
                   summary=False, text_file=None, **kwargs):
    if denoise is not None:
        denoised = denoise_audio(denoise)
        if mix:
            mix_audio(denoised, mix)

    if text_file is not None:
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()
        summary = summarise_text(text)
        print(summary)
        return
    if not any([url, filename]):
        raise ValueError("Either url or filename must be specified")

    if url is not None:
        # download video
        filename = download_video(url, "bestaudio" if audio_only else video_format)

    if split:
        # separate audio
        spleeter_separate(filename)

    filename = make_subtitles(device, filename, lang, model, output_format, prompt, subtitles, translate)

    if summary:
        with open(f"{filename}.txt", "r", encoding="utf-8") as f:
            text_file = f.read()
        summary = summarise_text(text_file)
        with open(f"{filename}.summary.txt", "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Summary written to {filename}.summary.txt")

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
        filename = filename.replace(Path(filename).suffix, "")
        create_srt_simple(filename, result["segments"], output_format)
        print(f"Subtitles written to {filename}")
    return filename


def create_srt_simple(filename, result, output_format):
    with open(f"{filename}.srt", "w", encoding="utf-8") as f:
        write_srt(result, f)
    with open(f"{filename}.txt", "w", encoding="utf-8") as f:
        write_txt(result, f)


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


def denoise_audio(filenames):
    denoised = []
    for filename in filenames:
        print("Denoising", filename, "...")
        sr, audio = wavfile.read(filename)
        # reduced_noise = nr.reduce_noise(y=audio, sr=sr)
        reduced_noise = audio
        wavfile.write(
            new_name := f"{Path(filename).stem}_denoised{Path(filename).suffix}",
            sr,
            reduced_noise,
        )
        denoised.append(new_name)
    return denoised


def mix_audio(filenames, output_filename):
    # List of input WAV files
    input_files = filenames
    print("Mixing", ", ".join(input_files), "to", output_filename, "...")

    # Read the first file to get the sampling rate and initialize the mixed data
    rate, mixed_data = wavfile.read(input_files[0])
    mixed_data = mixed_data.astype(np.int32)

    # Read and mix the remaining files
    for file in input_files[1:]:
        _, data = wavfile.read(file)
        mixed_data += data.astype(np.int32)

    # Average the mixed data
    mixed_data = mixed_data // len(input_files)

    # Ensure that the mixed samples are within the appropriate range
    if mixed_data.dtype == np.int16:
        mixed_data = np.clip(mixed_data, -32768, 32767)
    elif mixed_data.dtype == np.int32:
        mixed_data = np.clip(mixed_data, -2147483648, 2147483647)
    # Add more cases if needed (e.g., np.int8, np.int64)

    # Convert the mixed samples back to the original data type
    mixed_data = mixed_data.astype(np.int16)

    # Write the mixed samples to a new WAV file
    wavfile.write(output_filename, rate, mixed_data)


def main(**kwargs):
    start = time.time()
    speech_to_text(**kwargs)
    end = time.time()
    print(f"Time elapsed: {end - start:.2f} seconds")


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
    print("Done")
