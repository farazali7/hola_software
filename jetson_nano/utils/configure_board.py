from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets

def configure_board(serial_port):
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    board_type = BoardIds.CYTON_BOARD
    params.serial_port = serial_port

    board = BoardShim(board_type, params)
    emg_channels = board.get_emg_channels(board_type)

    return board, emg_channels
