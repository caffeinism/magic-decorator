# Simple demo function using LangChain

The only function implemented in this module is magic.

`magic` works as a decorator. It simply sends the function's signature and docstring to the LLM and expects to get a result.

It can be installed with `pip install magic-decorator` and used with `from magic_decorator import magic`.

```python
from langchain import SomeLLMModels
llm = SomeLLMModels()

from magic_decorator import magic
@magic(llm=llm)
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
@magic(llm=llm)
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
@magic(llm=llm)
def friend(message: str) -> str:
    """
    This function responds like a friend.
    """
friend("Hi, how are you today?")
```
```
'Hey! I am doing great, thank you. How about you?'
```

Added a decorator for langchain. In this case, it becomes LLMChain.
```python
llm = SomeLLMModels()
@magic_langchain(llm=llm)
def mychain(arg1: int, arg2: str):
    ...

mychain.predict(arg1=2, arg2="hi")
```

There is no preprocessing, no postprocessing, and no error handling in this module. It's just code to do a simple thing, so there are no prompts to encourage better answers.