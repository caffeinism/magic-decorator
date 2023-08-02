import inspect
import json
import logging
import textwrap
from functools import wraps

import openai
import sick_json
from pydantic import BaseModel, Field


def _function_stringfy(func):
    docstring = f'"""\n{inspect.cleandoc(inspect.getdoc(func))}\n"""'
    docstring = textwrap.indent(docstring, "    ")
    return f"def {func.__name__}{str(inspect.signature(func))}:\n" f"{docstring}"


JSON_PROMPT = "You should always answer according to the JSON schema below: "


def get_json_format_prompt(pydantic_model, default_prompt=JSON_PROMPT):
    return f"{default_prompt}\n" f"{pydantic_model.schema_json()}"


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


SYSTEM_PROMPT = (
    "You are now the following python function:\n"
    "```\n"
    "{function_code}\n"
    "```\n\n"
    "{format_instruction}"
)


def magic(return_all=False, **openai_kwargs):
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
                    "content": SYSTEM_PROMPT.format(
                        function_code=function_code,
                        format_instruction=json_prompt,
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

            if return_all:
                return bot_says
            else:
                return bot_says["return"]

        return do_magic

    return wrapper


try:
    from langchain.base_language import BaseLanguageModel
    from langchain.chains import LLMChain
    from langchain.output_parsers.pydantic import PydanticOutputParser
    from langchain.prompts import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        SystemMessagePromptTemplate,
    )
    from langchain.schema import BaseOutputParser

    class SickJsonOutputParser(PydanticOutputParser):
        return_all: bool = False

        def parse(self, text: str) -> dict:
            parsed = sick_json.parse(
                text,
                pydantic_model=self.pydantic_object,
            )
            return parsed if self.return_all else parsed["return"]

    def magic_langchain(llm: BaseLanguageModel, return_all=False):
        def wrapper(func):
            function_code = _function_stringfy(func)
            return_annotation = inspect.signature(func).return_annotation
            return_model = get_return_model(return_annotation)
            output_parser = SickJsonOutputParser(
                pydantic_object=return_model, return_all=return_all
            )

            system_prompt = (
                SYSTEM_PROMPT.format(
                    function_code=function_code,
                    format_instruction=output_parser.get_format_instructions(),
                )
                .replace("{", "{{")
                .replace("}", "}}")
            )

            argument_list = list(inspect.signature(func).parameters.keys())
            arguments = ", ".join(["{" + key + "}" for key in argument_list])
            user_prompt = f"{func.__name__}({arguments})"

            template = ChatPromptTemplate(
                input_variables=argument_list,
                messages=[
                    SystemMessagePromptTemplate.from_template(system_prompt),
                    HumanMessagePromptTemplate.from_template(user_prompt),
                ],
            )

            chain = LLMChain(
                llm=llm,
                prompt=template,
                output_parser=output_parser,
                return_final_only=True,
            )

            return chain

        return wrapper

except Exception as e:
    pass
