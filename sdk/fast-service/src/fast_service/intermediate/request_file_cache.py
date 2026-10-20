import os

from io import BytesIO

from ..file import FastServiceFile


class RequestFileCache:

    cache_dir: str

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir

        abs_path = os.path.abspath(self.cache_dir)
        if not os.path.exists(abs_path):
            os.makedirs(self.cache_dir)

    def cache_request_files(
        self, request_id: str, file_list: list[FastServiceFile]
    ) -> (str, list[FastServiceFile]):
        request_cache_dir = os.path.join(self.cache_dir, request_id)
        if not os.path.exists(request_cache_dir):
            os.makedirs(request_cache_dir)

        reopened_file_list: list[FastServiceFile] = []
        for file in file_list:
            cache_file_path = os.path.join(request_cache_dir, file.get_filename())
            input_file = file.get_file()
            with open(cache_file_path, "wb") as f:
                file_data = input_file.read()
                f.write(file_data)
                updated_io = BytesIO(file_data)
                updated_io.name = file.get_filename()
                reopened_file_list.append(FastServiceFile.load(updated_io))

        return request_cache_dir, reopened_file_list
