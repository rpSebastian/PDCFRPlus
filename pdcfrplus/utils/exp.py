import logging
import os

from sacred import Experiment
from sacred.observers import FileStorageObserver

from pdcfrplus.utils.utils import get_server_id

ex = Experiment("default")
logger = logging.getLogger("mylogger")
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s [%(levelname).1s] %(filename)s:%(lineno)d - %(message)s ",
    "%Y-%m-%d %H:%M:%S",
)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel("INFO")

ex.logger = logger


class ServerFileStorageObserver(FileStorageObserver):
    server_id = get_server_id()

    def _maximum_existing_run_id(self):
        dir_nrs = []
        for folder in os.listdir(self.basedir):
            if "-" in folder:
                a, b = folder.split("-")
                if int(a) == "server_id":
                    b = int(b)
                    dir_nrs.append(b)
        if dir_nrs:
            return max(dir_nrs)
        else:
            return 0

    def _make_run_dir(self, _id):
        os.makedirs(self.basedir, exist_ok=True)
        self.dir = None
        if _id is None:
            fail_count = 0
            _id = self._maximum_existing_run_id() + 1
            while self.dir is None:
                try:
                    self._make_dir(_id)
                except FileExistsError:  # Catch race conditions
                    if fail_count < 1000:
                        fail_count += 1
                        _id += 1
                    else:  # expect that something else went wrong
                        raise
        else:
            self.dir = os.path.join(self.basedir, "{}-{}".format(self.server_id, _id))
            os.mkdir(self.dir)

    def _make_dir(self, _id):
        new_dir = os.path.join(self.basedir, "{}-{}".format(self.server_id, _id))
        os.mkdir(new_dir)
        self.dir = new_dir  # set only if mkdir is successful
