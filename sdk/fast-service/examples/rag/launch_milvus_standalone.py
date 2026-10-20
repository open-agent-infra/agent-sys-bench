import os


def replace_sudo_in_file(file_path):
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


if __name__ == "__main__":
    milvus_dir = ".cache/milvus_standalone"
    os.makedirs(milvus_dir, exist_ok=True)
    filename = "milvus_standalone_embed.sh"
    file_path = os.path.join(milvus_dir, filename)

    download_cmd = f"curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o {file_path}"
    os.system(download_cmd)

    replace_sudo_in_file(file_path)

    start_cmd = f"cd {milvus_dir} && bash {filename} start"

    os.system(start_cmd)
