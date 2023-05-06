import inspect
import json
import textwrap
from functools import wraps

import openai


def _function_stringfy(func):
    docstring = f'"""\n{inspect.cleandoc(inspect.getdoc(func))}\n"""'
    docstring = textwrap.indent(docstring, "    ")
    return f"def {func.__name__}{str(inspect.signature(func))}:\n" f"{docstring}"


def magic(**openai_kwargs):
    def wrapper(func):
        @wraps(func)
        def do_magic(*args, **kwargs):
            function_code = _function_stringfy(func)
            arguments = []
            for arg in args:
                arguments.append(repr(arg))
            for key, value in kwargs.items():
                arguments.append(f"{key}={repr(value)}")
            arguments_string = f"{func.__name__}({', '.join(arguments)})"

            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are now the following python function:\n"
                        "```\n"
                        f"{function_code}\n"
                        f"```\n\n"
                        "Only respond with your `return` value."
                    ),
                },
                {
                    "role": "user",
                    "content": arguments_string,
                },
            ]

            response = openai.ChatCompletion.create(
                messages=messages,
                **openai_kwargs,
            )

            return eval(
                response.choices[0]
                .message.content.strip()
                .removeprefix("return")
                .strip()
            )

        return do_magic

    return wrapper
