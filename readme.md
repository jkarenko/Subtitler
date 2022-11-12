# Create subtitles for your videos with ease.
## Installation
### Requirements
- Python 3.10+
- ffmpeg
- pytorch-cuda
- OpenAI Whisper

### Install
```bash
pip install torch==1.13.0+cu116 -f https://download.pytorch.org/whl/cu116/
pip install git+https://github.com/openai/whisper.git 
```

## Usage
### Create a subtitle file from video
```bash
python subtitler.py
```

### Notes
- youtube-dl seems to be throttled by YouTube
- whisper's medium model is large, slow and requires a lot of memory
- there's repetition in the transcript for an unknown reason

## TODO
- [ ] Add a progress bar
- [ ] Add a way to specify the model
- [ ] Add a way to specify the language
- [ ] Add a way to specify the video
- [ ] Add a way to specify the output file
- [ ] Add a way to specify the output format
- [ ] Add a way to specify the output language


## Credits
- [OpenAI Whisper](https://github.com/openai/whisper)
- [youtube-dl](https://github.com/ytdl-org/youtube-dl/)
- [ffmpeg](https://ffmpeg.org/)
- [pytorch](https://pytorch.org/)
