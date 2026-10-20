from abc import ABCMeta, abstractmethod

from typing import Callable, Any


class RequestLoader(metaclass=ABCMeta):

    @abstractmethod
    def has_next(self) -> bool:
        pass

    @abstractmethod
    def next(self):
        pass

    @abstractmethod
    def close(self):
        pass


class SimpleRequestLoader(RequestLoader):

    line_to_request: Callable[[str], Any]
    file_path: str
    request_num: int
    buffer_size: int

    file = None

    request_buffer: list | None = None
    buffer_valid_size: int = 0

    buffer_index: int = 0

    sampling_params: dict = None

    def __init__(
        self,
        line_to_request: Callable[[str], Any],
        file_path: str,
        request_num: int,
        buffer_size: int = -1,
        sampling_params: dict = None,
    ):
        self.line_to_request = line_to_request
        self.file_path = file_path
        self.request_num = request_num
        self.buffer_size = buffer_size

        self.sampling_params = sampling_params
        if self.sampling_params is None:
            self.sampling_params = {}
        self.sampling_method = self.sampling_params.get("method", None)

        self.file = open(file_path, encoding="utf-8")
        total_records = 0
        for line in self.file:
            total_records += 1
        if self.request_num < 0:
            self.request_num = total_records

        self.reset()

    def reset(self, offset: int = 0):
        self.read_nlines = offset
        self.file.seek(offset)
        self.buffer_index = 0
        self.request_buffer = None
        self.buffer_valid_size = 0
        self.buffer_index = 0

        if self.sampling_method == "random":
            import numpy as np

            total_records = sum(1 for _ in open(self.file_path, encoding="utf-8"))
            self.sampled_indices = np.random.choice(
                range(total_records),
                max(self.request_num * 2, total_records),
                replace=False,
            )
        elif self.sampling_method == "uniform":
            import numpy as np

            total_records = sum(1 for _ in open(self.file_path, encoding="utf-8"))
            offset = self.sampling_params["offset"]
            step = self.sampling_params["step"]
            self.sampled_indices = np.arange(offset, total_records, step)[
                : self.request_num * 2
            ]
        else:
            self.sampled_indices = None

        self._get_next_batch()

    def _is_selected(self, rid: int) -> bool:
        if rid < 0:
            return False
        if self.sampling_method is None:
            return True
        else:
            return rid in self.sampled_indices

    def _get_next_batch(self):
        if self.buffer_size == -1:
            lines = self.file.readlines()
            self.request_buffer = [
                self.line_to_request(line)
                for i, line in enumerate(lines)
                if self._is_selected(i)
            ]
            if self.request_num != -1:
                self.request_buffer = self.request_buffer[: self.request_num * 2]
            self.buffer_valid_size = len(self.request_buffer)
        else:
            if self.request_buffer is None:
                self.request_buffer = [None] * self.buffer_size

            index = 0
            while index < self.buffer_size:
                line = self.file.readline()
                self.read_nlines += 1
                while not self._is_selected(self.read_nlines - 1):
                    line = self.file.readline()
                    self.read_nlines += 1
                    if line == "":
                        break

                if line == "":
                    break

                self.request_buffer[index] = self.line_to_request(line)
                index += 1
            self.buffer_valid_size = index

    def has_next(self) -> bool:
        # not read file yet
        if self.request_buffer is None:
            self._get_next_batch()

        # end of file
        if self.buffer_valid_size == 0:
            return False

        if self.buffer_size == -1:
            return self.buffer_index < self.buffer_valid_size
        else:
            if self.buffer_index < self.buffer_valid_size:
                return True

            # buffer run out, get next batch
            self._get_next_batch()
            if self.buffer_valid_size == 0:
                return False
            else:
                self.buffer_index = 0
                return True

    def next(self):
        if not self.has_next():
            raise Exception("No next request")
        result = self.request_buffer[self.buffer_index]
        self.buffer_index += 1
        return result

    def close(self):
        self.file.close()
