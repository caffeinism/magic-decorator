import inspect
import json
import logging
import textwrap
from functools import wraps
from typing import Awaitable, Callable

import sick_json
from langchain.base_language import BaseLanguageModel
from langchain.chains import LLMChain
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import BaseStringMessagePromptTemplate
from langchain.schema import BaseOutputParser, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

SYSTEM_PROMPT = (
    "You are now the following python function:\n"
    "```\n"
    "{function_code}\n"
    "```\n\n"
    "{format_instruction}"
)


def _function_stringfy(func):
    docstring = f'"""\n{inspect.cleandoc(inspect.getdoc(func))}\n"""'
    docstring = textwrap.indent(docstring, "    ")
    return f"def {func.__name__}{str(inspect.signature(func))}:\n" f"{docstring}"


def _get_return_model(return_annotation, with_thought: bool):
    if with_thought:

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

    else:

        class Answer(BaseModel):
            return_: return_annotation = Field(
                description=(
                    "The return value of the function."
                    " This value must always be in valid JSON format."
                ),
                alias="return",
            )

    return Answer


class _SickJsonOutputParser(PydanticOutputParser):
    return_all: bool = False

    def parse(self, text: str) -> dict:
        parsed = sick_json.parse(
            text,
            pydantic_model=self.pydantic_object,
        )
        return parsed if self.return_all else parsed["return"]


class ReprHumanMessagePromptTemplate(BaseStringMessagePromptTemplate):
    def format(self, **kwargs):
        text = self.prompt.format(**{key: repr(value) for key, value in kwargs.items()})
        return HumanMessage(content=text, additional_kwargs=self.additional_kwargs)


def _get_func_chain(
    llm: BaseLanguageModel, func: Callable, return_all: bool, with_thought: bool
):
    function_code = _function_stringfy(func)
    return_annotation = inspect.signature(func).return_annotation
    return_model = _get_return_model(return_annotation, with_thought)
    output_parser = _SickJsonOutputParser(
        pydantic_object=return_model, return_all=return_all
    )

    argument_list = list(inspect.signature(func).parameters.keys())
    arguments = ", ".join([f"{key}={{{key}}}" for key in argument_list])
    user_prompt = f"{func.__name__}({arguments})"

    template = ChatPromptTemplate(
        input_variables=argument_list,
        messages=[
            SystemMessage(
                content=SYSTEM_PROMPT.format(
                    function_code=function_code,
                    format_instruction=output_parser.get_format_instructions(),
                )
            ),
            ReprHumanMessagePromptTemplate.from_template(user_prompt),
        ],
    )

    chain = LLMChain(
        llm=llm,
        prompt=template,
        output_parser=output_parser,
        return_final_only=True,
    )

    return chain


def magic(llm: BaseLanguageModel, return_all=False, with_thought=False):
    def decorator(func: Callable):
        chain = _get_func_chain(
            llm=llm, func=func, return_all=return_all, with_thought=with_thought
        )
        signature = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            arguments = signature.bind(*args, **kwargs).arguments
            return chain.predict(**arguments)

        return wrapper

    return decorator


def magic_langchain(llm: BaseLanguageModel, return_all=False, with_thought=False):
    def decorator(func: Callable):
        return _get_func_chain(
            llm=llm, func=func, return_all=return_all, with_thought=with_thought
        )

    return decorator
