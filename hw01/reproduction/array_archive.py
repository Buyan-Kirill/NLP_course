"""NPZ serialization with platform-independent ZIP metadata."""
import zipfile
import numpy as np


def write_npz(path, arrays):
    """Match the original Windows NPZ bytes on every supported host."""
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for key, array in arrays.items():
            info = zipfile.ZipInfo(key + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 0
            info.external_attr = 0o600 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with archive.open(info, 'w', force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)
