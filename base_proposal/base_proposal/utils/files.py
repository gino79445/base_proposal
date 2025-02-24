import base_proposal
from pathlib import Path

# get paths
def get_root_path():
    path = Path(base_proposal.__path__[0]).resolve() / '..' / '..'
    return path


def get_urdf_path():
    path = get_root_path() / 'urdf'
    return path


def get_usd_path():
    path = Path(base_proposal.__path__[0]).resolve()/ 'usd'
    return path


def get_cfg_path():
    path = path = Path(base_proposal.__path__[0]).resolve()/ 'cfg'
    return path
