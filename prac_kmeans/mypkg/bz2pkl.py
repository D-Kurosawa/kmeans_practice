"""Compress and decompress pickle

compress:   convert pickle to bz2
decompress: convert bz2 to pickle
"""
import bz2
import pathlib
import pickle


def loads(compress_obj):
    try:
        decompress = bz2.decompress(compress_obj)
    except OSError:
        try:
            return pickle.loads(compress_obj)
        except Exception:
            raise IOError(f"object <{compress_obj}> is not pickle object")
    else:
        return pickle.loads(decompress)


def dumps(obj, compress_level=1, compress=True):
    pkl = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    if compress:
        return bz2.compress(pkl, compresslevel=compress_level)
    else:
        return pkl


def load(file_name):
    if not pathlib.Path(file_name).exists():
        raise FileNotFoundError(file_name)

    try:
        with bz2.BZ2File(file_name, 'rb') as f:
            pkl = f.read()
    except OSError:
        try:
            with open(file_name, 'rb') as f:
                return pickle.load(f)
        except Exception:
            raise IOError(f"file <{file_name}> can not read to pickle object")
    else:
        return pickle.loads(pkl)


def dump(obj, file_name, compress_level=1, compress=True):
    if compress:
        with bz2.BZ2File(file_name, 'wb', compresslevel=compress_level) as f:
            f.write(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    else:
        with open(file_name, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    pass
