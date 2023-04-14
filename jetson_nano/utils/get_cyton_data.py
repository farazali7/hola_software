def get_cyton_data(board, amount):
    data = board.get_current_board_data(amount) # get latest 256 packages or less, doesnt remove them from internal buffer
    # data = board.get_board_data()  # get all data and remove it from internal buffer

    return data
