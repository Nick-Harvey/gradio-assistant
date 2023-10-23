import re
import os
import sys
import shutil
import hashlib
from io import StringIO, BytesIO
from contextlib import contextmanager
from typing import List
from datetime import datetime, timedelta


class IO:
    @staticmethod
    def register(options):
        pass

    def open(self, path: str, mode: str):
        raise NotImplementedError

    def exists(self, path: str) -> bool:
        raise NotImplementedError

    def move(self, src: str, dst: str):
        raise NotImplementedError

    def copy(self, src: str, dst: str):
        raise NotImplementedError

    def makedirs(self, path: str, exist_ok=True):
        raise NotImplementedError

    def remove(self, path: str):
        raise NotImplementedError

    def listdir(self, path: str, recursive=False, full_path=False, contains=None):
        raise NotImplementedError

    def isdir(self, path: str) -> bool:
        raise NotImplementedError

    def isfile(self, path: str) -> bool:
        raise NotImplementedError

    def abspath(self, path: str) -> str:
        raise NotImplementedError

    def last_modified(self, path: str) -> datetime:
        raise NotImplementedError

    def md5(self, path: str) -> str:
        hash_md5 = hashlib.md5()
        with self.open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    re_remote = re.compile(r"(oss|https?)://")

    def islocal(self, path: str) -> bool:
        return not self.re_remote.match(path.lstrip())

class DefaultIO(IO):
    __name__ = 'DefaultIO'

    def _check_path(self, path):
        if not self.islocal(path):
            raise RuntimeError('fixthis')

    def open(self, path, mode='r'):
        self._check_path(path)
        path = self.abspath(path)
        return open(path, mode=mode)

    def exists(self, path):
        self._check_path(path)
        path = self.abspath(path)
        return os.path.exists(path)

    def move(self, src, dst):
        self._check_path(src)
        self._check_path(dst)
        src = self.abspath(src)
        dst = self.abspath(dst)
        shutil.move(src, dst)

    def copy(self, src, dst):
        self._check_path(src)
        self._check_path(dst)
        src = self.abspath(src)
        dst = self.abspath(dst)
        try:
            shutil.copyfile(src, dst)
        except shutil.SameFileError:
            pass

    def makedirs(self, path, exist_ok=True):
        self._check_path(path)
        path = self.abspath(path)
        os.makedirs(path, exist_ok=exist_ok)

    def remove(self, path):
        self._check_path(path)
        path = self.abspath(path)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

    def listdir(self, path, recursive=False, full_path=False, contains=None):
        self._check_path(path)
        path = self.abspath(path)
        contains = contains or ''
        if recursive:
            files = (os.path.join(dp, f) if full_path else f for dp, dn, fn in os.walk(path) for f in fn)
            files = [file for file in files if contains in file]
        else:
            files = os.listdir(path)
            if full_path:
                files = [os.path.join(path, file) for file in files if contains in file]
        return files

    def isdir(self, path):
        return os.path.isdir(path)

    def isfile(self, path):
        return os.path.isfile(path)

    def abspath(self, path):
        return os.path.abspath(path)

    def last_modified(self, path):
        return datetime.fromtimestamp(os.path.getmtime(path))


@contextmanager
def ignore_io_error(msg=""):

    try:
        yield
    except (oss2.exceptions.RequestError, oss2.exceptions.ServerError) as e:
        sys.stderr.write(str(e) + " " + msg)


@contextmanager
def mute_stderr():
    cache = sys.stderr
    sys.stderr = StringIO()
    try:
        yield None
    finally:
        sys.stderr = cache
