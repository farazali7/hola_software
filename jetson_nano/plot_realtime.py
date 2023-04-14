import logging

import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
from jetson_nano.utils import preprocess


class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.emg_channels = BoardShim.get_emg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title='BrainFlow Plot', size=(800, 600))

        self._init_timeseries()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtGui.QApplication.instance().exec_()

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        for i in range(len(self.emg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('TimeSeries Plot')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        data = np.transpose(data)
        preprocessed = preprocess(data)
        for count, channel in enumerate(self.emg_channels):
            # plot timeseries
            # DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            # DataFilter.perform_bandpass(data[channel], self.sampling_rate, 3.0, 100.0, 2,
            #                             FilterTypes.BUTTERWORTH.value, 0)
            # DataFilter.perform_bandstop(data[channel], self.sampling_rate, 48.0, 52.0, 2,
            #                             FilterTypes.BUTTERWORTH.value, 0)
            # DataFilter.perform_bandstop(data[channel], self.sampling_rate, 58.0, 62.0, 2,
            #                             FilterTypes.BUTTERWORTH.value, 0)

            self.curves[count].setData(preprocessed[..., channel].tolist())

        self.app.processEvents()


def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    serial_port = "/dev/ttyUSB0"
    params = BrainFlowInputParams()
    board_type = BoardIds.CYTON_BOARD
    params.serial_port = serial_port

    board_shim = BoardShim(board_type, params)
    try:
        board_shim.prepare_session()
        board_shim.start_stream(450000)
        Graph(board_shim)
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()


if __name__ == '__main__':
    main()