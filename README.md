# typingchat

A python style implementation of Typechat 

# Useage
```
pip3 install pytypingchat
```

local install
```
pip3 install .
```

## demo
Custom dataclass 
```
from typingchat.core import TypeChat,OpenAIChat
from pydantic import BaseModel,Field
from typing import Literal

class Customer(BaseModel):
    name: str = Field(description="Specify the customer's name")

class CoffeeDrink(BaseModel):
    type: Literal["CoffeeDrink"] = Field(default="CoffeeDrink", description="Specify the type of coffee drink")
    name: Literal["cappuccino", "flat white", "latte", "latte macchiato", "mocha", "chai latte"] = Field(None,                                                                                             description="Specify the name of the coffee drink")
    count: int = Field(1, description="Specify the selected drink quantity")

    def add_sugar(self, customer: Customer):
        """
        add some sugar to coffee

        Returns:

        """
        return "add sugar success"
```

```
OPENAI_API_KEY = "sk-xxx"
OPENAI_MODEL = "gpt-3.5-turbo-0125"

chat = OpenAIChat(api_key=OPENAI_API_KEY,model=OPENAI_MODEL)
typechat = TypeChat(chat=chat)
coffee: CoffeeDrink = typechat.translate( #type: ignore
    message="I want a mocha.",
    obj=CoffeeDrink
)

coffee.model_dump() # {'type': 'CoffeeDrink', 'name': 'mocha', 'count': 1})

```
output
```
{'type': 'CoffeeDrink', 'name': 'mocha', 'count': 1})
```
use as function call
```
chat = OpenAIChat(api_key=OPENAI_API_KEY, model=OPENAI_MODEL)
typechat = TypeChat(chat=chat)
product = typechat.translate(
    message="I want a flat white",
    obj=CoffeeDrink
)
filter = lambda name: True if name in [product.add_sugar.__name__] else False
reply_messsage = typechat.operate(product, message="Add a little sugar to Tom's coffee", func_filter=filter,extra_dataclass=[Customer])

reply_message.content # The sugar has been successfully added to Tom's coffee.
```


# References
1. https://github.com/microsoft/TypeChat