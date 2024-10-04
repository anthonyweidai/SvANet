import shutil

WinCols, _ = shutil.get_terminal_size()
WinCols = 80 if WinCols > 80 else WinCols