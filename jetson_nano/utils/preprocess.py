from classification.src.utils.preprocessing import butter_bandpass_filter, notch_filter, \
downsample

def preprocess(data):
    sampling_rate = 250  # Hz
    data = butter_bandpass_filter(data, 2, [10, 100], sampling_rate)
    
    data = notch_filter(data, 50, 30.0, sampling_rate)

    data = downsample(data, 200, sampling_rate)

    return data
