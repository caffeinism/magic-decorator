import inspect
import json
import logging
import textwrap
from functools import wraps

import sick_json
import jsonref
import openai
from pydantic import BaseModel, Field


def _function_stringfy(func):
    docstring = f'"""\n{inspect.cleandoc(inspect.getdoc(func))}\n"""'
    docstring = textwrap.indent(docstring, "    ")
    return f"def {func.__name__}{str(inspect.signature(func))}:\n" f"{docstring}"


JSON_PROMPT = "You should always answer according to the JSON schema below: "


def get_json_format_prompt(pydantic_model, default_prompt=JSON_PROMPT):
    return (
        f"{default_prompt}\n"
        f"{json.dumps(jsonref.loads(pydantic_model.schema_json()))}"
    )


def get_return_model(return_annotation):
    class Answer(BaseModel):
        thought: str = Field(
            description="Write down your thoughts or reasoning step by step."
        )
        return_: return_annotation = Field(
            description=(
                "The return value of the function."
                " This value must always be in valid JSON format."
            ),
            alias="return",
        )

    return Answer


def magic(return_thought=False, **openai_kwargs):
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

            return_annotation = inspect.signature(func).return_annotation
            return_model = get_return_model(return_annotation)
            json_prompt = get_json_format_prompt(return_model)

            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are now the following python function:\n"
                        "```\n"
                        f"{function_code}\n"
                        f"```\n\n"
                        f"{json_prompt}"
                    ),
                },
                {
                    "role": "user",
                    "content": arguments_string,
                },
            ]
            
            
            logging.debug("System Message: ")
            logging.debug(messages[0]["content"])
            logging.debug("User Message: ")
            logging.debug(messages[1]["content"])

            response = openai.ChatCompletion.create(
                messages=messages,
                **openai_kwargs,
            )
            
            logging.debug("Bot Message: ")
            logging.debug(response.choices[0].message.content)

            bot_says = sick_json.parse(
                response.choices[0].message.content,
                pydantic_model=return_model,
            )
                
            if return_thought:
                return bot_says["return"], bot_says["thought"]
            else:
                return bot_says["return"]

        return do_magic

    return wrapper
