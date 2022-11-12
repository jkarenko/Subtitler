from youtube_dl import YoutubeDL
import whisper
import datetime
import pathlib

url = "https://youtu.be/hqnIGzR8jU0"

# download media, audio only
audio_dl = YoutubeDL({'format': 'best'})
audio_dl.extract_info(url)
audio_file_path = audio_dl.prepare_filename(audio_dl.extract_info(url))

# load pre-trained whisper model and send it to GPU, transcribe audio to text
model = whisper.load_model("medium", device="cuda").to("cuda")
result = whisper.transcribe(model=model, language="Finnish", audio=audio_file_path)

# extract inferred transcript and format it as SRT subtitles
srt = ""
for i, s in enumerate(result["segments"]):
    seq = s["id"] + 1
    start = str.replace(str(datetime.timedelta(seconds=float(s["start"]))), ".", ",")
    end = str.replace(str(datetime.timedelta(seconds=float(s["end"]))), ".", ",")
    if "," not in start: start += ",000"
    if "," not in end: end += ",000"
    text = s["text"].strip().split(" ")
    # if len(text) > 4: // uncomment to break long lines
    #     text.insert(len(text) // 2 + 1, "\n")
    text = " ".join(text).replace("\n ", "\n")
    srt += f"{seq}\n{start} --> {end}\n{text}\n\n"

# save SRT subtitles to file
with open(f"{audio_file_path.replace(pathlib.Path(audio_file_path).suffix, '.srt')}", "w") as f:
    f.write(srt)
