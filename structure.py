# view_structure.py
import os

def generate_project_tree(start_path='.', ignore_dirs=None, ignore_files=None):
    """
    Generates and prints a tree-like diagram of a project directory.

    Args:
        start_path (str): The root directory to start the scan from.
        ignore_dirs (set): A set of directory names to ignore.
        ignore_files (set): A set of file names to ignore.
    """
    # Set default ignored directories if none are provided
    if ignore_dirs is None:
        ignore_dirs = {'.git', '__pycache__', '.vscode', 'venv', '.idea'}

    # Set default ignored files, including this script itself
    if ignore_files is None:
        ignore_files = {os.path.basename(__file__), '.gitignore'}

    # Get the name of the root directory to print first
    project_root_name = os.path.basename(os.path.abspath(start_path))
    print(f"{project_root_name}/")

    # Start the recursive generation of the tree
    _recursive_tree_builder(start_path, "", ignore_dirs, ignore_files)

def _recursive_tree_builder(directory, prefix, ignore_dirs, ignore_files):
    """
    Recursively builds and prints the directory tree.
    """
    try:
        # Get all items in the current directory, filtering out ignored ones
        items = [item for item in os.listdir(directory) if item not in ignore_dirs and item not in ignore_files]
        items.sort()
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory}")
        return
    except PermissionError:
        print(f"Error: Permission denied for directory {directory}")
        return

    # Define the tree branch characters
    pointers = ['├── '] * (len(items) - 1) + ['└── ']

    for pointer, item_name in zip(pointers, items):
        full_path = os.path.join(directory, item_name)
        print(prefix + pointer + item_name)

        if os.path.isdir(full_path):
            # Determine the prefix for the next level of the tree
            # If the current item is the last one, the prefix should be empty space
            extension = '│   ' if pointer == '├── ' else '    '
            # Recurse into the subdirectory
            _recursive_tree_builder(full_path, prefix + extension, ignore_dirs, ignore_files)


if __name__ == "__main__":
    # The script will scan the directory it is placed in.
    # To run, place this file in your project's root folder and execute:
    # python view_structure.py
    generate_project_tree('.')
