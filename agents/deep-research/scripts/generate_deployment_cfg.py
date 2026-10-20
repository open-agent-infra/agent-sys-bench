import os
import ast
import csv
from pathlib import Path


def extract_fast_service_functions(directory, module_prefix):
    """
    Extract all functions decorated with @xxx.fast_service from Python files in the given directory.
    Extract both sync functions and async functions.
    Returns a list of tuples (module_name, function_name, filepath).
    """
    fast_service_functions = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                print(f"Processing file: {filepath}")

                # Calculate module name from file path
                # Remove the directory prefix and .py extension
                relative_path = os.path.relpath(filepath, directory)
                module_name = relative_path[:-3].replace(
                    os.path.sep, "."
                )  # Remove .py and convert path separators to dots
                # Ensure the module name starts with the specified prefix
                module_name = (
                    f"{module_prefix}.{module_name}" if module_prefix else module_name
                )

                # Parse the Python file
                with open(filepath, "r", encoding="utf-8") as f:
                    try:
                        tree = ast.parse(f.read(), filename=file)
                    except Exception as e:
                        print(f"Could not parse {filepath}: {e}")
                        continue

                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Check for @xxx.fast_service decorator
                            for decorator in node.decorator_list:
                                if (
                                    isinstance(decorator, ast.Attribute)
                                    and isinstance(decorator.value, ast.Name)
                                    and decorator.attr == "fast_service"
                                ):
                                    function_name = node.name
                                    fast_service_functions.append(
                                        (module_name, function_name, filepath)
                                    )
                                    print(
                                        f"Found sync function: {module_name}.{function_name} in {filepath}"
                                    )
                                    break

                        # Check for async functions
                        if isinstance(node, ast.AsyncFunctionDef):
                            for decorator in node.decorator_list:
                                if (
                                    isinstance(decorator, ast.Attribute)
                                    and isinstance(decorator.value, ast.Name)
                                    and decorator.attr == "async_fast_service"
                                ):
                                    function_name = node.name
                                    fast_service_functions.append(
                                        (module_name, function_name, filepath)
                                    )
                                    print(
                                        f"Found async function: {module_name}.{function_name} in {filepath}"
                                    )
                                    break

    return fast_service_functions


def save_to_csv(data, output_file):
    """
    Save the extracted function data to a CSV file.
    """
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["module_name", "function_name", "filepath"])
        writer.writerows(data)

    print(f"Results saved to {output_file}")


def generate_server_config(data, port, output_file):
    """
    Generate a server configuration file in yaml from the extracted function data.
    Example:
    ```yaml
        service_impl: fastapi
        ip: 0.0.0.0
        port: {port}
        service_list:
        - module_name: {module_name}
            name: {function_name}
            ip: 0.0.0.0
            port: {port}

        monitor: True
        monitor_storage:
            type: csv
            store_dir: ./.cache/output/monitor/server
    ```
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("service_impl: fastapi\n")
        f.write(f"ip: 0.0.0.0\n")
        f.write(f"port: {port}\n")
        f.write("service_list:\n")
        for module_name, function_name, filepath in data:
            f.write(f"  - module_name: {module_name}\n")
            f.write(f"    name: {function_name}\n")
            f.write(f"    ip: 0.0.0.0\n")
            f.write(f"    port: {port}\n")
        f.write("monitor: True\n")
        f.write("monitor_storage:\n")
        f.write("  type: csv\n")
        f.write("  store_dir: ./.cache/output/monitor/server\n")

    print(f"Server config generated and saved to {output_file}")


def generate_client_config(data, port, server_ip, server_port, output_file):
    """
    Generate a client configuration file in yaml from the extracted function data.
    Example:
    ```yaml
        client_mode: True
        service_impl: fastapi
        ip: 0.0.0.0
        port: {port}
        service_list:
        - module_name: {module_name}
            name: {function_name}
            ip: {server_ip}
            port: {server_port}

        monitor: True
        monitor_storage:
            type: csv
            store_dir: ./.cache/output/monitor/client
    ```
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("client_mode: True\n")
        f.write("service_impl: fastapi\n")
        f.write(f"ip: 0.0.0.0\n")
        f.write(f"port: {port}\n")
        f.write("service_list:\n")
        for module_name, function_name, filepath in data:
            f.write(f"  - module_name: {module_name}\n")
            f.write(f"    name: {function_name}\n")
            f.write(f"    ip: {server_ip}\n")
            f.write(f"    port: {server_port}\n")
        f.write("monitor: True\n")
        f.write("monitor_storage:\n")
        f.write("  type: csv\n")
        f.write("  store_dir: ./.cache/output/monitor/client\n")

    print(f"Server config generated and saved to {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract @xxx.fast_service functions from Python files"
    )
    parser.add_argument("-d", "--directory", help="Directory to scan for Python files")
    parser.add_argument(
        "--server_ip",
        default="agent_server",
        help="IP address for server config (default: legoAgent-server)",
    )
    parser.add_argument(
        "--server_port", default=20000, help="Port for server config (default: 20000)"
    )
    parser.add_argument(
        "--client_port", default=20000, help="Port for client config (default: 20000)"
    )
    parser.add_argument(
        "--module_prefix",
        default="agsbench_deep_research.agents.anthropic.skills",
        help="Module prefix for function names",
    )

    parser.add_argument(
        "--output_dir",
        default="./.cache",
        help="Output CSV file path (default: ./.cache)",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return

    functions = extract_fast_service_functions(args.directory, args.module_prefix)
    print(f"Found {len(functions)} @fast_service functions")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    functions_path = Path(args.output_dir) / "fast_service_functions.csv"
    save_to_csv(functions, functions_path)
    generate_server_config(
        functions, args.server_port, Path(args.output_dir) / "server.yaml"
    )
    generate_client_config(
        functions,
        args.client_port,
        args.server_ip,
        args.server_port,
        Path(args.output_dir) / "client.yaml",
    )


if __name__ == "__main__":
    main()
