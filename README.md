# GenAIProjects

A file system agent powered by OpenAI's Responses API (Azure OpenAI). The agent uses tool-calling to perform local file operations based on natural language commands.

## Tools

| Tool | Description |
|------|-------------|
| `read_file` | Reads a file and returns its content with metadata (name, size, extension) |
| `list_files` | Lists all files in a directory with metadata |
| `write_file` | Writes content to a file |
| `search_file` | Case-insensitive keyword search within a file, returns matching lines |

## Setup

1. Install dependencies:
   ```bash
   pip install openai
   ```

2. Set environment variables:
   ```bash
   export AZURE_OPENAI_ENDPOINT_CA="<your-azure-openai-endpoint>"
   export AZURE_OPENAI_KEY_CA="<your-azure-openai-key>"
   ```

3. Run:
   ```bash
   python fs_tools.py
   ```
