import os.path as osp
import os
from pathlib import Path

PROJECT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))


def _data_dir():
    """Use the containing research project's data directory when present."""
    override = os.environ.get('ACAC_DATA_DIR')
    if override:
        return str(Path(override).expanduser().resolve())
    for parent in Path(PROJECT_DIR).resolve().parents:
        if (parent / 'PROJECT_SPEC.md').is_file() and (parent / 'AGENTS.md').is_file():
            return str(parent / 'data')
    return osp.join(PROJECT_DIR, 'data')


DATA_DIR = _data_dir()


