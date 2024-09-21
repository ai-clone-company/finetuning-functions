import json
from datetime import datetime, timedelta
from typing import Literal

from loguru import logger
from pydantic import BaseModel


class Message(BaseModel):
    date: datetime
    author: str
    text: str


class Chat(BaseModel):
    name: str
    type: Literal["personal_chat", "private_group", "private_supergroup"]
    messages: list[Message]
    sessions: list[list[Message]] = []


def load_chats(path: str) -> tuple[list[Chat], tuple[int | None, str | None]]:
    chats: list[Chat] = []
    target_id, target_name = None, None
    logger.info(f"Loading chats from '{path}'...")
    with open(path, "r") as f:
        for chat in json.load(f)["chats"]["list"]:
            # It means we encountered 'Saved Messages', from which we can extract id and a name of a target person
            if "name" not in chat:
                target_id = int(chat["id"])
                target_name = str(
                    next(
                        msg
                        for msg in chat["messages"]
                        if msg["from_id"] == f"user{target_id}"
                    )["from"]
                )
            # If chat does not contain name that means we encountered "Deleted Account"
            elif chat["name"]:
                messages = [
                    Message(
                        date=msg["date"],
                        author=msg["from"],
                        text="".join(
                            [
                                text_entity["text"]
                                for text_entity in msg["text_entities"]
                            ]
                        )
                        + msg.get("sticker_emoji", ""),
                    )
                    for msg in chat["messages"]
                    if "from" in msg
                    and msg["from"]
                    and (msg["text_entities"] or "sticker_emoji" in msg)
                ]
                if messages:
                    chat = Chat(name=chat["name"], type=chat["type"], messages=messages)
                    chats.append(chat)
    logger.info(f"Found {len(chats)} chats in file '{path}'")
    if not target_name:
        logger.warning("Was not able to detect target name from 'Saved Messages'!")
    return chats, (target_id, target_name)


def transform_chats(
    input: str,
    output: str,
    target_name: str | None = None,
    last_x_months: int = 60,
    session_minutes_threshold: int = 10,
    concat_one_user_messages_delimeter: str = "\n>>>",
):
    chats, (_, extracted_target_name) = load_chats(input)
    target_name = target_name or extracted_target_name
    logger.info(f"Preparing dataset for user with name '{target_name}'...")

    cutoff_date = datetime.now() - timedelta(days=last_x_months * 30)
    chats = [
        chat
        for chat in chats
        if (
            filtered_messages := [
                msg for msg in chat.messages if msg.date > cutoff_date
            ]
        )
        and setattr(chat, "messages", filtered_messages)
    ]
    logger.info(f"After filtering by date, there are {len(chats)} chats left")

    for chat in chats:
        chat.sessions = create_sessions(chat.messages, session_minutes_threshold)
        chat.sessions = combine_consecutive_messages(
            chat.sessions, concat_one_user_messages_delimeter
        )
        chat.sessions = [
            session
            for session in chat.sessions
            if any(msg.author == target_name for msg in session[1:])
        ]

    all_sessions = [session for chat in chats for session in chat.sessions]
    session_dicts = [
        {
            "text": "\n".join(
                f"<|im_start|>{msg.author}\n{msg.text}<|im_end|>" for msg in session
            )
        }
        for session in all_sessions
    ]

    with open(output, "w", encoding="utf-8") as f:
        for session in session_dicts:
            json.dump(session, f, ensure_ascii=False)
            f.write("\n")
    logger.info(f"Took {len(all_sessions)} chat sessions and wrote them to '{output}'.")


def create_sessions(messages: list[Message], threshold: int) -> list[list[Message]]:
    sessions = []
    current_session = []
    for msg in messages:
        if (
            not current_session
            or (msg.date - current_session[-1].date).seconds / 60 < threshold
        ):
            current_session.append(msg)
        else:
            sessions.append(current_session)
            current_session = [msg]
    if current_session:
        sessions.append(current_session)
    return sessions


def combine_consecutive_messages(
    sessions: list[list[Message]], delimiter: str
) -> list[list[Message]]:
    combined_sessions = []
    for session in sessions:
        combined_session = []
        current_message = session[0]
        current_message.text = delimiter.lstrip() + current_message.text
        for msg in session[1:]:
            if msg.author == current_message.author:
                current_message.text += delimiter + msg.text
            else:
                combined_session.append(current_message)
                current_message = msg
                current_message.text = delimiter.lstrip() + current_message.text
        combined_session.append(current_message)
        combined_sessions.append(combined_session)
    return combined_sessions


if __name__ == "__main__":
    transform_chats("./data/result.json", "./data/training_data.jsonl")
