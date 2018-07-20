import wave
import tkinter as tk
from tkinter import filedialog
import struct
import numpy
from pydub import AudioSegment

# Remove tkinter gui for file open dialogs
root = tk.Tk
root.withdraw


def wav_to_floats(wave_file):
    """Parses a wave file to a float array

        Args:
            wave_file: File path of wave file to open

        Returns:
            A float array representing the audio

    """
    # with wave.open(wave_file) as w:
    #     num_frames = w.getnframes()
    #     num_channels = w.getnchannels()
    #     astr = w.readframes(num_frames)
    #     a = struct.unpack("%ih" % (num_frames * num_channels), astr)
    #     a = [float(val) / pow(2, 15) for val in a]
    #
    #     if num_channels is not 1:
    #         a = a

    audio = AudioSegment.from_wav(wave_file)

    if audio.channels is not 1:
        audio = audio.set_channels(1)

    if audio.frame_rate is not 44100:
        audio = audio.set_frame_rate(44100)

    if audio.sample_width is not 2:
        audio = audio.set_sample_width(2)

    audio = audio.get_array_of_samples()
    audio = [float(val) / pow(2, 15) for val in audio]

    return audio


def wav_to_array_dialog():
    """Gets the path of wave file with popup dialog, parses it to a float array and returns it

        Returns:
            A float array representing the audio

    """
    audio_file_path = filedialog.askopenfilename()
    audio_array = numpy.array(wav_to_floats(audio_file_path))
    return audio_array


def save_wav_file_from_array(audio_array, sample_rate):
    """Saves a float array as wave file with popup dialog

        Args:
            audio_array: Array of floats representing audio to be saved
            sample_rate: The desired sampling rate to save audio with

    """
    from scipy.io.wavfile import write
    save_path = get_save_path()
    write(save_path, sample_rate, audio_array)


def get_save_path():
    """Gets a save path with popup dialog

        Returns:
            The desired save file path

    """
    return filedialog.asksaveasfilename()

