import DaVinciResolveScript as dvr_script
import argparse
import subtitler
import time

# class Task:
#     render_job = None
#     render_audio = None
#     subtitles = None
#     media = None


resolve = dvr_script.scriptapp("Resolve")
audio_output_path = r"v:\video"
input_video_path = r"c:\code\python\Subtitler\Maailman vesipäivä - Pohjavesi-hqnIGzR8jU0.mp4"
pm = resolve.GetProjectManager()
# project = pm.GetCurrentProject()
# timeline = project.GetCurrentTimeline()
# mediapool = project.GetMediaPool()

task = dict(render_job=None, render_audio=None, subtitles=None, media=None, project=None, timeline=None, status=0,
            completion=0, audio_path=None, mediapool=None)


# def create_new_project(project_name, timeline_name):
#     project = pm.CreateProject(project_name)
#     timeline = project.GetMediaPool().CreateEmptyTimeline(timeline_name)
#     subtitles = mediapool.ImportMedia(r"subtitle_file.srt")
#     mediapool.AppendToTimeLine(media)
#     mediapool.AppendToTimeLine(subtitles)


def add_media_to_timeline(t):
    mediapool = t["project"].GetMediaPool()
    mediapool.AppendToTimeline(mediapool.ImportMedia(t["subtitles"]))


def render_audio(t):
    settings = {
        "SelectAllFrames": True,
        "TargetDir": audio_output_path,
        "CustomName": t["timeline"].GetName(),
        "UniqueFilenameStyle": 1,
        "ExportVideo": False,
        "ExportAudio": True,
        "AudioCodec": "mp3",
        "AudioBitDepth": 16,
        "AudioSampleRate": 22050,
    }
    t["project"].SetRenderSettings(settings)
    t["render_job"] = t["project"].AddRenderJob()
    t["project"].StartRendering()
    return t


def check_if_render_complete(t):
    t["completion"] = t["project"].GetRenderJobStatus(t["render_job"])["CompletionPercentage"]
    t["status"] = t["project"].GetRenderJobStatus(t["render_job"])["JobStatus"]
    return t


def create_subtitles(t):
    return subtitler.speech_to_text(filename=t["audio_path"], model="medium", lang="en", output_format="srt")


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project", help="Project name")
    parser.add_argument("-t", "--timeline", help="Timeline name")
    return parser.parse_args()


def main(t):
    t = render_audio(t)
    print("Rendering audio...")
    while t["completion"] < 100:
        t = check_if_render_complete(t)
        print("Render: " + str(t["completion"]) + "%")
        time.sleep(5)
    print("Render " + str(t["status"]))

    task["audio_path"] = t['project'].GetRenderJobs(task['render_job'])[1]['TargetDir'] + "\\" + \
                         t['project'].GetRenderJobs(t['render_job'])[1]['OutputFilename']
    t["project"].DeleteRenderJob(t["render_job"])
    print(f"File saved to {t['audio_path']}")
    print("Creating subtitles")
    t["subtitles"] = create_subtitles(t)
    print("Adding subtitles to timeline")
    add_media_to_timeline(t)
    print("Done")


if __name__ == "__main__":
    print("Starting Subtitler")
    args = arg_parse()
    if args.project:
        task["project"] = pm.CreateProject(args.project)
        task["timeline"] = task["project"].GetMediaPool().CreateEmptyTimeline(args.timeline)
    else:
        task["project"] = pm.GetCurrentProject()
        task["timeline"] = task["project"].GetCurrentTimeline()
        print(f"Using open project \"{task['project'].GetName()}\" and timeline \"{task['timeline'].GetName()}\"")
    main(task)
