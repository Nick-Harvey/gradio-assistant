import dataclasses
from enum import auto, Enum
from typing import List, Tuple
import os
import logging
from langchain.memory import ConversationBufferMemory   

class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()

@dataclasses.dataclass    
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "\n "
    sep2: str = None

    skip_next: bool = False

    def get_prompt(self):
        self.system = "The following is a conversation between a curious human and AI. The AI gives helpful, detailed, and polite answers to the human's questions."
        self.sep = "\n"
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _ = message
                    ret += role.replace("AI", "AI") + ": " + message + self.sep
                else:
                    if role != "":
                        ret += role.replace("AI", "AI") + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    # def to_gradio_chatbot(self):
    #     ret = []
    #     for i, (role, msg) in enumerate(self.messages[self.offset :]):
    #         if i % 2 == 0:
    #             ret.append([msg, None])
    #         else:
    #                 ret[-1][-1] = msg
    #         return ret

    # def to_gradio_chatbot(self):
    #     # Extract messages from chat_memory object
    #     ret = []
    #     for message in self.messages:
    #         content = message.content
    #         if message.__class__.__name__ == 'HumanMessage':
    #             ret.append([content, None])
    #         elif message.__class__.__name__ == 'AIMessage':
    #             # Ensure there's a previous message to append the response to.
    #             if ret:
    #                 ret[-1][-1] = content
    #             else:
    #                 # Handle the case where the first message is an AI message.
    #                 # Adjust as per your needs.
    #                 ret.append([None, content])

    #     return ret
    def to_gradio_chatbot(self):
    # Extract messages from the state's messages attribute
        ret = []
        for role, content in self.messages:
            if role == 'HumanMessage':
                ret.append([content, None])
            elif role == 'AI':
                # Ensure there's a previous message to append the response to.
                if ret:
                    ret[-1][-1] = content
                else:
                    # Handle the case where the first message is an AI message.
                    # Adjust as per your needs.
                    ret.append([None, content])

        return ret
        

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


snkl_assistant_convo = Conversation(
    system="The following is a conversation between a curious human and assistant AI. The assistant AI gives helpful, detailed, and polite answers to the human's questions.",
    roles=("HumanMessage", "AI"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

default_conversation = snkl_assistant_convo

if __name__ == "__main__":
    print(default_conversation.get_prompt())
