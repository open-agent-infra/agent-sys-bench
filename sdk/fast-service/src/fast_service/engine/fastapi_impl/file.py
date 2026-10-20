from typing import BinaryIO
from fastapi import UploadFile

from ...file import FastServiceFile


class FastApiFile(FastServiceFile):

    file: UploadFile

    def __init__(self, file: UploadFile) -> None:
        self.file = file

    def get_file(self) -> BinaryIO:
        return self.file.file

    def get_header(self) -> dict[str, str]:
        return dict(self.file.headers.items())

    def get_size(self) -> int:
        return self.file.size

    def get_filename(self) -> str:
        return self.file.filename

    def content_type(self) -> str | None:
        return self.file.content_type

    def write(self, data: bytes) -> None:
        self.file.write(data)

    async def read(self, size: int = -1) -> bytes:
        return await self.file.read(size)

    def seek(self, offset: int) -> None:
        self.file.seek(offset)

    def close(self) -> None:
        self.file.close()
