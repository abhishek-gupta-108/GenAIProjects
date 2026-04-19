import asyncio
import os

async def read_file(file_path):
    """Takes in an absolute file path and returns the content of the file along with metadata such as file name, size, and extension."""
    with open(file_path, 'r') as file:
        content = file.read()
        metadata = {
            'file_name': file_path,
            'size': len(content),
            'extension': f'{file_path.split(".")[-1]}'
        }
    return (content, metadata)


async def list_files(directory):
    """Takes in a directory path and returns a list of files in the directory along with their metadata such as file name, size, and extension."""
    files = []
    for entry in os.scandir(directory):
        if entry.is_file():
            files.append({
                'file_name': entry.name,
                'size': entry.stat().st_size,
                'extension': f'{entry.name.split(".")[-1]}'
            })
    return files


async def write_file(file_path, content):
    """Takes in an absolute file path and content, writes the content to the file, and returns a success message."""
    with open(file_path, 'w') as file:
        file.write(content)
    return True


async def search_file(file_path, keyword):
    """Takes in an absolute file path and a keyword, searches the file in a case-insensitive way, and returns the lines containing the keyword."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matching_lines = [line for line in lines if keyword.lower() in line.lower()]
    return matching_lines


async def get_tools():
    tools = [
            {
                "type": "function",
                "name": "read_file",
                "description": read_file.__doc__,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The absolute path to the file to be read."
                        }
                    },
                    "required": ["file_path"]
                }
            },
            {
                "type": "function",
                "name": "list_files",
                "description": list_files.__doc__,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "The path to the directory to list files from."
                        }
                    },
                    "required": ["directory"]
                }
            },
            {
                "type": "function",
                "name": "write_file",
                "description": write_file.__doc__,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The absolute path to the file to be written."
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to be written to the file."
                        }
                    },
                    "required": ["file_path", "content"]
                }
            },
            {
                "type": "function",
                "name": "search_file",
                "description": search_file.__doc__,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The absolute path to the file to be searched."
                        },
                        "keyword": {
                            "type": "string",
                            "description": "The keyword to search for in the file."
                        }
                    },
                    "required": ["file_path", "keyword"]
                }
            }
        ]
    
    tools_map = {
        "read_file": read_file,
        "list_files": list_files,
        "write_file": write_file,
        "search_file": search_file
    }
    return (tools, tools_map)
