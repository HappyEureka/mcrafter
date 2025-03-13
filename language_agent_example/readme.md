create a new file:
- credentials.py to include the following:

1. absolute path to the project folder.
2. openai client.

Note: it is important to use an api that supports structured output.

Eaxmple:

```python
from openai import AzureOpenAI
mcrafter_abs_path = ...

client = AzureOpenAI(
  azure_endpoint = ...,
  api_key= ...,  
  api_version= ..."
)