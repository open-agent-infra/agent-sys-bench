import os
import shutil

from typing import Optional, Any

from .record import IntermediateRecord, IntermediateRequest, IntermediateResponse


class FunctionFile:

    function_name: str

    function_dir: str
    file_path: str
    file: Any

    writing_count: int

    def __init__(self, function_name: str, file_dir: str, file_name: str):
        self.function_name = function_name

        self.function_dir = os.path.join(file_dir, function_name)
        abs_dir_path = os.path.abspath(self.function_dir)
        if not os.path.exists(abs_dir_path):
            os.mkdir(abs_dir_path)

        file_path = os.path.join(self.function_dir, file_name)
        self.file_path = file_path
        self.file = open(
            self.file_path,
            "a",
            encoding="utf-8",
        )

        self.writing_count = 0

    def write(self, record: IntermediateRecord):
        self._write(record.dump_record())
        if record.has_file():
            self._save_request_files(record.request_id, record.get_cache_dir())

    def _write(self, line_of_data: str):
        self.file.write(f"{line_of_data}\n")

        self.writing_count += 1
        if self.writing_count % 10 == 0:
            self.flush()

    def _save_request_files(self, request_id: str, cache_dir: str):
        request_dir = os.path.join(self.function_dir, request_id)
        abs_path = os.path.abspath(request_dir)
        if not os.path.exists(abs_path):
            os.mkdir(abs_path)
        else:
            raise FileExistsError(f"Dir {abs_path} already exists.")

        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(abs_path, file)
                shutil.move(src_path, dst_path)
        shutil.rmtree(cache_dir)

    def flush(self):
        self.file.flush()
        os.fsync(self.file.fileno())

        self.writing_count = 0

    def close(self):
        self.file.close()


class FunctionFileSet:
    req_file: FunctionFile
    resp_file: FunctionFile

    def __init__(self, function_name: str, file_dir: str):
        self.req_file = FunctionFile(function_name, file_dir, "requests.txt")
        self.resp_file = FunctionFile(function_name, file_dir, "responses.txt")

    def flush(self):
        self.req_file.flush()
        self.resp_file.flush()

    def close(self):
        self.req_file.close()
        self.resp_file.close()


class FileManager:

    dir_path: str
    file_set_dict: dict[str, FunctionFileSet]

    def __init__(self, dir_path: str):
        self.dir_path = dir_path
        self.file_set_dict = {}

        abs_dir_path = os.path.abspath(dir_path)
        if not os.path.exists(abs_dir_path):
            os.mkdir(abs_dir_path)

    def register_function(self, function_name: str):
        if function_name in self.file_set_dict:
            return

        file_set = FunctionFileSet(function_name, self.dir_path)
        self.file_set_dict[function_name] = file_set

    def _get_file_set(self, function_name: str) -> FunctionFileSet:
        if function_name not in self.file_set_dict:
            self.register_function(function_name)

        return self.file_set_dict[function_name]

    def write_req(self, record: IntermediateRequest):
        file_set: FunctionFileSet = self._get_file_set(record.function_name)
        file_set.req_file.write(record)

    def write_resp(self, record: IntermediateResponse):
        file_set: FunctionFileSet = self._get_file_set(record.function_name)
        file_set.resp_file.write(record)

    def flush(self):
        for file_set in self.file_set_dict.values():
            file_set.flush()

    def close(self):
        for file_set in self.file_set_dict.values():
            file_set.close()
