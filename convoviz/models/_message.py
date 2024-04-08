"""Represents a single message in a conversation. It's contained in a Node object.

object path : conversations.json -> conversation -> mapping -> mapping node -> message
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional, Union, List

from pydantic import BaseModel, ConfigDict, Field, validator

from convoviz.utils import DEFAULT_MESSAGE_CONFIGS, MessageConfigs, code_block

if TYPE_CHECKING:
    from datetime import datetime

AuthorRole = Literal["user", "assistant", "system", "tool"]
class ImageAssetPointer(BaseModel):
    content_type: str
    asset_pointer: str
    size_bytes: int
    width: int
    height: int
    fovea: Any
    metadata: Any

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, values):
        # Serialize metadata if it's a dictionary
        if isinstance(values.get('metadata'), dict):
            values['metadata'] = json.dumps(values['metadata'])
        elif values.get('metadata') is None:
            values['metadata'] = "{}"
        # Ensure fovea is a string, even if it's None
        if values.get('fovea') is None:
            values['fovea'] = ""
        return values


class MessageAuthor(BaseModel):
    """Type of the `author` field in a `message`."""

    role: AuthorRole
    name: str | None = None
    metadata: dict[str, Any]


class MessageContent(BaseModel):
    """Type of the `content` field in a `message`."""

    content_type: str
    parts: List[Union[str, ImageAssetPointer]] = []
    text: str | None = None
    result: str | None = None
    
    @validator('parts', pre=True, each_item=True)
    def validate_parts(cls, v, values, **kwargs):
        if isinstance(v, dict):
            return ImageAssetPointer(**v)
        elif isinstance(v, str):
            return v
        else:
            raise ValueError(f"Invalid type for parts: {type(v)}")


class MessageMetadata(BaseModel):
    """Type of the `metadata` field in a `message`."""

    model_slug: str | None = None
    invoked_plugin: dict[str, Any] | None = None
    is_user_system_message: bool | None = None
    user_context_message_data: dict[str, Any] | None = None

    model_config = ConfigDict(protected_namespaces=())


class Message(BaseModel):
    """Wrapper class for the `message` field in a `node`.

    see `MessageJSON` and `models.Node` for more details
    """

    __configs: ClassVar[MessageConfigs] = DEFAULT_MESSAGE_CONFIGS

    id: str  # noqa: A003
    author: MessageAuthor
    create_time: datetime | None = None
    update_time: datetime | None = None
    content: MessageContent
    status: str
    end_turn: bool | None = None
    weight: float
    metadata: MessageMetadata
    recipient: str

    @classmethod
    def update_configs(cls, configs: MessageConfigs) -> None:
        """Set the configuration for all messages."""
        cls.__configs.update(configs)

    @property
    def header(self) -> str:
        """Get the title header of the message based on the configs."""
        return self.__configs["author_headers"][self.author.role]

    @property
    def text(self) -> str:
        """Get the text content of the message."""
        if self.content.parts is not None:
            return str(self.content.parts[0])
        if self.content.text is not None:
            return code_block(self.content.text)
        if self.content.result is not None:
            return self.content.result

        # this error caught some hidden bugs in the data. need more of these
        err_msg = f"No valid content found in message: {self.id}"
        raise ValueError(err_msg)
