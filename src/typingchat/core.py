import json
from typing import TypeVar, Union
from pydantic import BaseModel
from .chat import ChatMessage, OpenAIChat
from pydantic import ValidationError
from typing import Literal, List, Any, Callable
from pydantic import Field
import inspect
from typing import Dict

T = TypeVar('T')


class FunctionCall(BaseModel):
    name: str = Field(description="Specify the function name to be called")
    arguments: str = Field(default={},
                           description="Specifies the parameters of the function call, JSON characters converted from a dictionary of type python")


class ToolCall(BaseModel):
    type: Literal["function"] = "function"
    functions: List[FunctionCall] = Field(default=[],
                                          description="Specify function call object, If you need multiple functions, please sort them in order.")


def default_func_filter(name: str) -> bool:
    """
    Default class filter function

    Args:
        name: Specify function name

    Returns:
        bool: true or false

    """
    if not name.startswith("_"):
        return True
    else:
        return False


def gen_function_call_code(func: Callable, func_name: str = None):
    """
    Generate descriptions about function calls

    Args:
        func:
        func_name:

    Returns:

    """

    if not func_name:
        func_name = func.__name__
    func_signature = inspect.signature(func)
    func_doc = "\n".join([f"    {line}" for line in inspect.getdoc(func).splitlines()])
    func_code = F"""
def {func_name}{func_signature}:
    \"\"\"
{func_doc}
    \"\"\""""
    return func_code


def gen_dataclass_code(dataclass: Any):
    """
    Generate data class code

    Args:
        func:
        func_name:

    Returns:

    """
    cls_codes = []
    clean_space_num = 0
    for line in inspect.getsource(dataclass).splitlines():
        if "class " in line:
            clean_space_num = len(line) - len(line.strip())

        if line.strip().startswith("def "):
            break
        if clean_space_num:
            cls_codes.append(line.replace(" " * clean_space_num, ""))
        else:
            cls_codes.append(line)
    code = "\n".join(cls_codes)
    return code


class ExecResult(BaseModel):
    func: str
    result: Any
    status: Literal["fail", "success"]


class TypeChat:
    def __init__(self, chat: OpenAIChat, instruction: str = None):
        """

        Args:
            chat: `OpenAIChat`
            instruction: Custom instruction

        Examples:
            >>> class Customer(BaseModel):
            ...    name: str = Field(description="Specify the customer's name")
            >>> class CoffeeDrink(BaseModel):
            ...    type: Literal["CoffeeDrink"] = Field(default="CoffeeDrink", description="Specify the type of coffee drink")
            ...    name: Literal["cappuccino", "flat white", "latte", "latte macchiato", "mocha", "chai latte"] = Field(None,                                                                                             description="Specify the name of the coffee drink")
            ...    count: int = Field(1, description="Specify the selected drink quantity")
            ...
            ...    def add_sugar(self, customer: Customer):
            ...        '''
            ...        add some sugar to coffee
            ...
            ...        Returns:
            ...
            ...        '''
            ...        return "add sugar success"
            >>> import os
            >>> OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
            >>> OPENAI_MODEL = os.environ.get("OPENAI_MODEL","gpt-3.5-turbo-0125")
            >>> chat = OpenAIChat(api_key=OPENAI_API_KEY,model=OPENAI_MODEL)
            >>> typechat = TypeChat(chat=chat)
            >>> coffee: CoffeeDrink = typechat.translate( #type: ignore
            ... message="I want a mocha.",
            ... obj=CoffeeDrink
            >>> coffee.model_dump()
            {'type': 'CoffeeDrink', 'name': 'mocha', 'count': 1}

            # function call
            >>> typechat = TypeChat(chat=chat)
            >>> product = typechat.translate(
            >>> message="I want a flat white",
            ...    obj=CoffeeDrink
            ...    )
            >>> filter = lambda name: True if name in [product.add_sugar.__name__] else False
            >>> reply_messsage = typechat.operate(product, message="Add a little sugar to Tom's coffee", func_filter=filter,extra_dataclass=[Customer])
            >>> reply_messsage.content
            The sugar has been successfully added to Tom's coffee.

        )
        """
        self.chat = chat
        self.instruction = instruction
        self.current_operation_obj = None

    def generate_user_prompt(self, message: str, type_name: str, schema: Union[dict, str], with_chat: bool) -> str:
        """
        Generate user prompt words

        Args:
            message: input info
            type_name: type name
            schema: object json shema
            schema_type: Literal["dataclass"]

        Returns:

        """
        if type(schema) == dict:
            schema = json.dumps(schema, indent=4, ensure_ascii=False)

        if with_chat:
            prompt = f"""
The following is a my request:
'''
{message}
'''

You need to return the results to me in JSON objects JSON objects of type "{type_name}" according to the following JSON Schema definitions:
```
{schema}
```
Return JSON object with 2 spaces of indentation.
"""
        else:

            prompt = f"""
You are a service that translates user requests into JSON objects of type "{type_name}" according to the following JSON Schema definitions:
```
{schema}
```
The following is a user request:
'''
{message}
'''
The following is the user request translated into a JSON object with 2 spaces of indentation.
"""

        return prompt

    def extract_json(self, content: str) -> dict:
        """
        extract json from reply content

        Args:
            content: special json content

        Returns:
            dict: return json obj

        Raises:
            json.decoder.JSONDecodeError: Json text parsing error
            ValueError: Response did not contain any text resembling JSON.

        """

        first_curly = content.find("{")
        last_curly = content.rfind("}") + 1
        if 0 <= first_curly < last_curly:
            trimmed_response = content[first_curly:last_curly]
            parsed_response = json.loads(
                trimmed_response
            )
            return parsed_response
        else:
            raise ValueError(
                "Response did not contain any text resembling JSON."
            )

    def _generate_repair_prompt(self, error_message: str) -> str:
        prompt = f"""
The JSON program object is invalid for the following reason:
'''
{error_message}
'''
The following is a revised JSON program object:
"""
        return prompt

    def generate_toolcall_sys_message(self, obj_name, funcs: Dict[str, Callable], extra_dataclass: List[Any] = None):
        toolcall_schema = json.dumps(ToolCall.model_json_schema(), indent=4, ensure_ascii=False)
        codes = []
        for func_name, call in funcs.items():
            func_code = gen_function_call_code(
                func=call,
                func_name=func_name
            )
            codes.append(func_code)

        func_code_schema = "\n".join(codes)

        cls_codes = []
        if extra_dataclass:
            for dataclass in extra_dataclass:
                cls_code = gen_dataclass_code(dataclass)
                cls_codes.append(cls_code)

        prompt = f"""
You are a service that can operate python classes. The current class instance is `{obj_name}`, which has the following callable functions, The function definition is as follows:

```
{func_code_schema}
```

"""
        if cls_codes:
            prompt += f"""
The data classes currently used are as follows:
```
{'\n'.join(cls_codes)}
```
"""

        prompt += f"""
When a function call is required, your output conforms to the json result of the following json schema
```
{toolcall_schema}
```
"""

        return prompt

    def exec_tool_call(self, reply_message: ChatMessage, funcs: Dict[str, Callable]):
        exec_results: List[ExecResult] = []
        tool = self.poll_message_request(
            reply_message=reply_message,
            obj=ToolCall,
            max_error_count=3
        )
        toolcall: ToolCall = tool  # type: ignore

        for func in toolcall.functions:
            exec_obj = funcs[func.name]
            params = json.loads(func.arguments, strict=False)
            try:
                exec_result = exec_obj(**params)
                exec_results.append(ExecResult(
                    func=func.name,
                    result=exec_result,
                    status="success"
                ))
            except Exception as e:
                exec_results.append(ExecResult(
                    func=func.name,
                    result=str(f"exec error: {str(e)}"),
                    status="fail"
                ))

        return exec_results

    def operate(self, obj: Any, message: str,
                func_filter: Callable[[str], bool] = default_func_filter,
                extra_funcs: Dict[str, Callable] = None,
                extra_dataclass: List[Any] = None,
                ) -> ChatMessage:
        """

        Args:
            obj:
            message:
            func_filter:
            extra_funcs:
            extra_dataclass:

        Returns:

        """
        if not inspect.isclass(type(obj)):
            raise ValueError("The object is not a class.")

        funcs: Dict[str, Callable] = {}
        for func in inspect.getmembers(obj, predicate=inspect.ismethod):
            func_name = func[0]
            call = func[1]
            if func_filter(func_name):
                funcs[func_name] = call
        if extra_funcs:
            funcs.update(extra_funcs)

        if obj != self.current_operation_obj:
            self.chat.history.clear()
            if funcs:
                obj_name = type(obj).__name__
                sys_message = self.generate_toolcall_sys_message(obj_name, funcs, extra_dataclass)
                if self.instruction:
                    sys_message += f"\n# INSTRUCTIONS\n {self.instruction}"

                self.chat.history.insert(
                    0, ChatMessage(
                        content=sys_message,
                        role="system"
                    )
                )

        reply_message = self.chat.prompt(
            message=message,
            role="user"
        )
        while True:
            if '"type": "function"' in reply_message.content:
                exec_results = self.exec_tool_call(reply_message, funcs=funcs)
                if exec_results:
                    reply_message = self.chat.prompt(
                        "\n".join([f"exec status: {exec_result.status}\nresult: {exec_result.result}" for exec_result in
                                   exec_results])
                    )
            else:
                break
        return reply_message

    def poll_message_request(self, reply_message: ChatMessage, obj: T, max_error_count: int = 5) -> T:
        """

        Args:
            reply_message: poll message
            obj:
            max_error_count:

        Returns:

        Raises:
            ValueError: No object matched

        """
        current_error_count = 0
        error_message = None
        result = None
        while True:
            if current_error_count > max_error_count:
                raise ValueError(f"Maximum number of errors exceeded, final error message is {error_message}")
            if error_message:
                reply_message = self.chat.prompt(
                    message=self._generate_repair_prompt(error_message=error_message), role="user"
                )
            content = reply_message.content
            try:
                json_obj = self.extract_json(content=content)
                result = obj.model_validate(obj=json_obj)
                error_message = ""
            except ValidationError as e:
                error_message = str(e)
            except json.decoder.JSONDecodeError as e:
                error_message = str(e)
            except ValueError as e:
                error_message = str(e)
            finally:
                if error_message: current_error_count += 1
            if not error_message:
                break

        if not result:
            raise ValueError("No object matched")
        return result

    def translate(self, message: str, obj: T, max_error_count: int = 5, with_chat: bool = False,
                  schema: dict = None) -> T:
        """
        Convert natural language into data objects

        Args:
            message: natural language input
            obj: Specail pydantic BaseModel
            max_error_count: Specifies the maximum number of errors for callback processing
            with_chat:

        Returns:
            T: Output type

        Raises:
            ValueError: Currently only Pydantic’s BaseModel data type is supported
        """

        if not with_chat:
            self.chat.history.clear()
        if type(obj) != type(BaseModel):
            raise ValueError(
                "Currently only Pydantic’s BaseModel data type is supported"
            )

        obj: BaseModel

        if self.instruction:
            self.chat.history.append(
                ChatMessage(content=self.instruction, role="user")
            )

        if not schema:
            schema = obj.model_json_schema(),

        prompt_message = self.generate_user_prompt(
            message=message,
            type_name=obj.__name__,
            schema=schema,
            with_chat=with_chat
        )

        reply_message = self.chat.prompt(
            message=prompt_message,
            role="user"
        )

        result = self.poll_message_request(
            reply_message=reply_message,
            max_error_count=max_error_count,
            obj=obj
        )
        return result
