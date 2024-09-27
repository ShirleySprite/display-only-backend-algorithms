from pathlib import Path

from backend_algorithms.import_model.import_utils.import_dataset import ImportAVDataset
from backend_algorithms.utils.av import run_subprocess, get_audio_info


def to_cbr(
        input_file,
        output_file,
        bit_rate,
        channels,
        sample_rate
):
    return run_subprocess(
        ['ffmpeg', '-i', str(input_file), "-b:a", bit_rate, "-ac", str(channels), "-ar", str(sample_rate), "-y",
         str(output_file)]
    )


def get_conventional_bitrate(input_bitrate):
    input_bitrate /= 1000

    conventional_bitrates = ['64k', '80k', '96k', '112k', '128k', '144k', '160k', '192k', '224k', '256k', '320k']

    for bitrate in conventional_bitrates:
        if int(bitrate[:-1]) >= input_bitrate:
            return bitrate

    return conventional_bitrates[-1]


def trans(input_json):
    origin_path = Path(input_json.get('originPath'))
    target_path = Path(input_json.get('targetPath'))

    for x in ImportAVDataset(origin_path, "*.mp3"):
        sample_rate, channels, bit_rate = get_audio_info(
            x.meta_path, stream="sample_rate, bit_rate, channels"
        ).split('x')
        target_bitrate = get_conventional_bitrate(int(bit_rate))

        mp3_output = target_path / x.meta_path.relative_to(origin_path)
        mp3_output.parent.mkdir(parents=True, exist_ok=True)
        to_cbr(
            x.meta_path,
            mp3_output,
            target_bitrate,
            channels,
            sample_rate
        )
