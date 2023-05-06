# Simple demo function using OpenAI's ChatComplete

The only function implemented in this module is magic.

`magic` works as a decorator. It simply sends the function's signature and docstring to the GPT and expects to get a result.

It can be installed with `pip install magic-decorator` and used with `from magic_decorator import magic`.

```python
from magic_decorator import magic
@magic(model="gpt-4")
def transpose(matrix: list[list[int]]) -> list[list[int]]:
    """
    This function finds the transpose of the given matrix.
    """
transpose([[1,2,3],[4,5,6]])
```
```
[[1, 4], [2, 5], [3, 6]]
```

```python
from magic_decorator import magic
@magic(model="gpt-4")
def add(a: float, b: float) -> float:
    """
    This function adds a and b.
    """
add(1., 10.)
```
```
11.0
```

```python
from magic_decorator import magic
@magic(model="gpt-4")
def friend(message: str) -> str:
    """
    This function responds like a friend.
    """
friend("Hi, how are you today?")
```
```
'Hey! I am doing great, thank you. How about you?'
```

Since the magic decorator works with OpenAI's API, you need to put the API_KEY in an environment variable or directly in the decorator.
```python
import os
from magic_decorator import magic

os.environ["OPENAI_API_KEY"] = ...
# or
@magic(model="gpt-4", api_key=...)
def func():
    ...
```
Alternatively, you can pre-declare decorators that will be used repeatedly.
```python
cold_magic = magic(model="gpt-4", temperature=0)
@cold_magic
def func():
    ...
```

There is no preprocessing, no postprocessing, and no error handling in this module. It's just code to do a simple thing, so there are no prompts to encourage better answers.