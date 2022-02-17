import librosa 
import numpy as np
#1 audio -> np array
#2 create hamming windows
#3 do fft on buffers
#4 subtract noise spectrum
#5 go back to time domain using ifft
#6 overlap & save buffer, adding samples to the partially overlapped buffers
def get_mp3_file(mp3_file):
    return {mp3_file.split('/')[-1]:AudioSegment.from_mp3(mp3_file)}

def load_audio(files,num_processors=num_processors):
    files.sort()
    p = Pool(processes = 28)
    output = p.map(get_mp3_file,files)
    p.close()
    audio_dict = {k:v for i in output for k,v in i.items()}
    return audio_dict

def pydub_to_np(audio):
    return np.array(audio.get_array_of_samples(), dtype=np.float64).reshape((-1, audio.channels)).T / (1<<(8*audio.sample_width)), audio.frame_rate

def np_to_hamming(np_array):
    frame_len, hop_len = 16, 8
    frames = librosa.util.frame(np_array, frame_length=frame_len, hop_length=hop_len)
    windowed_frames = np.hanning(frame_len).reshape(-1, 1)*frames
    return windowed_frames

def fft_on_windows(windowed_frames):
    fft_arr = []
    for i in range(len(windowed_frames)):
        fft_arr.append(np.fft.fft(windowed_frames[i]))
    return fft_arr

def ifft_on_fft_arr(fft_array):
    ifft_arr = []
    for i in range(len(fft_array)):
        ifft_arr.append(np.fft.ifft(fft_array[i]))
    return ifft_arr

def overlap(ifft_array):
    frame_len, hop_len, sampling_len = 16, 8, 512
    librosa.frames_to_time(frames, 512, hop_len)

