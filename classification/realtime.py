from openbci_stream.acquisition import OpenBCIConsumer, Cyton
from pylsl import StreamInlet, resolve_stream
import time


if __name__ == "__main__":
    # with OpenBCIConsumer(mode='serial', endpoint='/dev/tty.usbserial-DM00Q2K0', streaming_package_size=200, daisy=False) \
    #         as (stream, openbci):
    #     for i, message in enumerate(stream):
    #         if message.topic == 'emg':
    #             print(message.value['samples'])

    # openbci = Cyton('serial', '/dev/tty.usbserial-DM00Q2K0', capture_stream=True)
    # openbci.start_stream()
    # time.sleep(10)
    # openbci.end_stream()
    # print('d')

    # resolve an EMG stream on the lab network and notify the user
    print("Looking for an EMG stream...")
    streams = resolve_stream('type', 'EMG')
    inlet = StreamInlet(streams[0])
    print("EMG stream found!")

    # initialize thresholds and variables for storing time
    prev_time = 0
    flex_thres = 1.0

    while True:
        sample, timestamp = inlet.pull_sample()  # get EMG data_processing sample and its timestamp

        if any(x > 0 for x in sample):
            print(sample)

        if (((sample[1] >= flex_thres) or (sample[0] >= flex_thres))):  # if an EMG peak is detected from any of the arms

            if (sample[1] > sample[0]):
                print('here 1')
            else:
                print('here 2')
