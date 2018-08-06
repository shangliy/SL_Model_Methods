"""
Dir related functions
"""
import os

def create_dirs(dirs):
    """Create a list of dir.

    Args:
        dirs: (list) a list of directories

    Returns:
        exit_code: (bool) 0 success; -1 fail
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.mkdir(dir_)
        return 0
    except Exception as err:
        print("Creaste directories error : {0}".format(err))
        exit(-1)
