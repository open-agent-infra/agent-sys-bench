import os, sys
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Launch VDB")
    parser.add_argument("--vdb_type", type=str, default="chroma")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--name", type=str, default="VectorDB")
    parser.add_argument("--home", type=str, default="./.cache/VectorDB")
    parser.add_argument("--sudo", action="store_true")

    return parser.parse_args()


def launch_chroma(args):
    port = args.port
    name = args.name
    cmd = f"docker run --rm -d -p {port}:{port} --name {name} chromadb/chroma"
    os.system(cmd)


def remove_sudo_in_file(file_path):
    try:
        # Read the content of the file
        with open(file_path, "r") as file:
            content = file.read()

        # Replace "sudo " with ""
        new_content = content.replace("sudo ", "")

        # Write the modified content back to the file
        with open(file_path, "w") as file:
            file.write(new_content)

        print(f"Successfully replaced 'sudo ' with '' in {file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


def launch_milvus(args):
    milvus_home = args.home
    os.makedirs(milvus_home, exist_ok=True)
    filename = "milvus_standalone_embed.sh"
    file_path = os.path.join(milvus_home, filename)

    download_cmd = f"curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o {file_path}"
    os.system(download_cmd)

    if not args.sudo:
        remove_sudo_in_file(file_path)

    start_cmd = f"cd {milvus_home} && bash {filename} start"

    os.system(start_cmd)


if __name__ == "__main__":
    args = get_args()
    if args.vdb_type == "chroma":
        launch_chroma(args)
    elif args.vdb_type == "milvus":
        launch_milvus(args)
    else:
        print("Unsupported VDB type")
        sys.exit(1)
