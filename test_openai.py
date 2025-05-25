import json
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Тестовый запрос с function calling
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {'role': 'system', 'content': 'You must call the test function'},
        {'role': 'user', 'content': 'Please call the function'}
    ],
    tools=[{
        'type': 'function',
        'function': {
            'name': 'test_function',
            'description': 'Test function',
            'parameters': {
                'type': 'object',
                'properties': {
                    'value': {'type': 'string'}
                },
                'required': ['value']
            }
        }
    }],
    tool_choice='required'
)

print('Content:', repr(response.choices[0].message.content))
print('Tool calls:', bool(response.choices[0].message.tool_calls))
if response.choices[0].message.tool_calls:
    print('Function arguments:', response.choices[0].message.tool_calls[0].function.arguments) 