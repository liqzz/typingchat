from typing import List
from openai import OpenAI
from typing import Literal
from pydantic import BaseModel
from typing import Optional

DEFAULT_BASE_API_URL = "https://api.openai.com/v1/"

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]  # 执行角色
    content: Optional[str]

    def to_message(self) -> dict:
        return self.model_dump(
            include={"role", "content"}
        )


class OpenAIChat:
    def __init__(self,
                 api_key: str,
                 base_url: str = DEFAULT_BASE_API_URL,
                 openai_kwargs: dict = None,
                 system_message: str = None,
                 model: str = None
                 ):
        """
        OpenAI chat

        Args:
            api_key: special openai api key
            base_url: openai api url
            openai_kwargs: openai params
            system_message: system message
            model: model name

        Examples:
            >>> chat = OpenAIChat(api_key="sk-OTgoVisw6e...")
            >>> message = chat.prompt(message="hello", model="gpt-3.5-turbo-0125")
            >>> message.model_dump()
            {'role': 'assistant', 'content': 'Hello! How can I assist you today?'}

        """
        self.openai_kwargs = openai_kwargs or {"timeout": 600, "max_retries": 2}
        self.openai_client = OpenAI(api_key=api_key, base_url=base_url, **self.openai_kwargs)
        self.chat = self.openai_client.chat
        self._system_message = system_message
        self.history: List[ChatMessage] = []
        self.model = model
        if self._system_message:
            self.history.insert(0, ChatMessage(role="system", content=self.system_message))

    @property
    def system_message(self):
        return self._system_message

    @system_message.setter
    def system_message(self, message: str):
        self._system_message = message
        chat_message = ChatMessage(role="system", content=self.system_message)
        if not self.history:
            self.history.append(chat_message)
        else:
            self.history[0] = chat_message

    def prompt(self, message: str,
               role: Literal["system", "user", "assistant"] = "user",
               model: str = "gpt-3.5-turbo-0125",
               chat_kwargs: dict = None
               ) -> ChatMessage:
        """
        chat with ai

        Args:
            message: user input message
            role: message rike
            model: chat model
            chat_kwargs: chat extra params

        Returns:

        """
        chat_kwargs = chat_kwargs or {}
        messages = self.history
        if isinstance(message, ChatMessage):
            self.history.append(message)
        else:
            messages.append(ChatMessage(role=role, content=message))

        model = self.model or model
        completion = self.chat.completions.create(
            messages=[item.to_message() for item in messages],
            model=model,
            **chat_kwargs
        )
        ai_message = completion.choices[0].message
        reply_message = ChatMessage(
            role=ai_message.role,
            content=ai_message.content
        )

        messages.append(reply_message)
        return reply_message
