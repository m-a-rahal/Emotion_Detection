from pathlib import Path
import os


def create_path(path, is_file=True):
    """
    creates path leading to a file
        is_file: default = True, if the path is a folder set this to False
                if True only the parent folder is created (to avoid creating a folder having the file's name)
    """
    path = Path(path)
    # if path is a file, create the parent folder only
    if is_file:
        path = path.parent
    # ---- recursive call --------------------------
    if not os.path.exists(path.parent):
        create_path(path.parent, use_parent=False)
    # ---- stop conditions -------------------------
    if str(path) == '':
        return False
    if not os.path.exists(path):
        os.mkdir(path)
    return True
