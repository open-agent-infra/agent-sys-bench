import os
import subprocess
import sys


def generate_proto():
    """Generate Python code from proto file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    proto_file = os.path.join(current_dir, "fast_service.proto")

    # Command to generate Python code from proto file
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={current_dir}",
        f"--python_out={current_dir}",
        f"--grpc_python_out={current_dir}",
        proto_file,
    ]

    subprocess.check_call(cmd)
    print(f"Generated Python code from {proto_file}")


if __name__ == "__main__":
    generate_proto()
