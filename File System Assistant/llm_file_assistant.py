import fs_tools
import os
import json
import asyncio
from openai import OpenAI

async def run_agent():

    user_input = input("Enter your command: ")
    base_url = os.environ.get("AZURE_OPENAI_ENDPOINT_CA", "")
    api_key = os.environ.get("AZURE_OPENAI_KEY_CA", "")

    client = OpenAI(
        base_url = base_url,
        api_key = api_key,
    )
    tools, tools_map = await fs_tools.get_tools()
    
    responses = client.responses.create(
        model="gpt-4o",
        tools = tools,
        input = [
            {
                "role": "system",
                "content": "You are an assistant for file system operations. You can read files, list files in a directory, write to files, and search for keywords in files. Use the provided tools to perform these operations based on the user's commands."
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
    )

    for response in responses.output:
        if response.type == "function_call":
            tool_name = response.name
            tool_args = json.loads(response.arguments)
            if tool_name in tools_map:
                tool_function = tools_map[tool_name]
                tool_result = await tool_function(**tool_args)
                print(f"Tool: {tool_name}, Arguments: {tool_args}, Result: {tool_result}")
            else:
                print(f"Tool {tool_name} not found in tools map.")




if __name__ == "__main__":
    asyncio.run(run_agent())
