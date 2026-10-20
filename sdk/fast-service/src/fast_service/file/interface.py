import os

from abc import ABC, abstractmethod
from typing import BinaryIO, Optional

from ..description import FunctionDescriptor, ParamDescriptor


class FastServiceFile(ABC):

    @abstractmethod
    def get_file(self) -> BinaryIO:
        pass

    @abstractmethod
    def get_header(self) -> dict[str, str]:
        pass

    @abstractmethod
    def get_size(self) -> int:
        pass

    @abstractmethod
    def get_filename(self) -> str:
        pass

    @abstractmethod
    def content_type(self) -> str | None:
        pass

    @abstractmethod
    def write(self, data: bytes) -> None:
        pass

    @abstractmethod
    def read(self, size: int = -1) -> bytes:
        pass

    @abstractmethod
    def seek(self, offset: int) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"filename={self.get_filename()!r}, "
            f"size={self.get_size()!r}, "
            f"headers={self.get_header()!r})"
        )

    @classmethod
    def load(cls, file: BinaryIO) -> "FastServiceFile":
        return NaiveFastServiceFile(file, filename=os.path.basename(file.name))


class FileParamUsage:
    f: FunctionDescriptor

    descriptor: Optional[ParamDescriptor | None]

    def __init__(self, f: FunctionDescriptor):
        self.f = f

        descriptor_list = f.cls_dict.get(FastServiceFile)

        if descriptor_list:
            if len(descriptor_list) > 1:
                raise TypeError(
                    "FastServiceFiles param is repeated defined in given function."
                )
            self.descriptor = descriptor_list[0]
        else:
            self.descriptor = None

    def use_file(self) -> bool:
        return self.descriptor is not None

    def file_param_name(self) -> str:
        return self.descriptor.name

    def single_file(self) -> bool:
        return not self.descriptor.is_list

    def get_file_arg(
        self, *args, **kwargs
    ) -> FastServiceFile | list[FastServiceFile] | None:
        if len(args) > self.descriptor.index:
            return args[self.descriptor.index]
        elif self.descriptor.name in kwargs:
            return kwargs[self.descriptor.name]
        else:
            return None

    def rewrite_file_arg(self, updated_file_arg, *args, **kwargs) -> (tuple, dict):
        if len(args) > self.descriptor.index:
            updated_args = []
            for i in range(len(args)):
                if i == self.descriptor.index:
                    updated_args.append(updated_file_arg)
                else:
                    updated_args.append(args[i])
            updated_args = tuple(updated_args)
        else:
            updated_args = args
            kwargs[self.descriptor.name] = updated_file_arg

        return updated_args, kwargs


class NaiveFastServiceFile(FastServiceFile):
    file: BinaryIO
    filename: str
    size: int
    header: dict[str, str]

    def __init__(
        self,
        file: BinaryIO,
        *,
        size: int | None = None,
        filename: str | None = None,
        header: dict[str, str] | None = None,
    ) -> None:
        self.filename = filename
        self.file = file
        self.size = size
        self.header = header or {}

    def get_file(self) -> BinaryIO:
        return self.file

    def get_filename(self) -> str:
        return self.filename

    def get_size(self) -> int:
        return self.size

    def get_header(self) -> dict[str, str]:
        return self.header

    def content_type(self) -> str | None:
        return self.header.get("content-type", None)

    def write(self, data: bytes) -> None:
        if self.size is not None:
            self.size += len(data)

        self.file.write(data)

    def read(self, size: int = -1) -> bytes:
        return self.file.read(size)

    def seek(self, offset: int) -> None:
        self.file.seek(offset)

    def close(self) -> None:
        self.file.close()
