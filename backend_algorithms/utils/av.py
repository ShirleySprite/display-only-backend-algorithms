import subprocess


def run_subprocess(
        popenargs
):
    try:
        result = subprocess.run(
            popenargs,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()

    except subprocess.CalledProcessError as e:
        return f"Error: {e}"


def get_audio_info(
        input_path,
        stream='width, height'
):
    return run_subprocess(
        ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', f'stream={stream}', '-of', 'csv=s=x:p=0',
         input_path]
    )


def get_video_info(
        input_path,
        stream='width, height'
):
    return run_subprocess(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', f'stream={stream}', '-of', 'csv=s=x:p=0',
         input_path]
    )
