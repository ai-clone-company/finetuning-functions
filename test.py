
from dotenv import load_dotenv

load_dotenv()

# print(os.getenv("OPENAI_API_KEY"))

import marvin


@marvin.fn
def truncate(text: str) -> str:
    """
    Truncates a conversation history presented in a special chat format without breaking the format.
    The purpose is to keep the conversation history short enough to be processed by GPT-3.5-turbo without
    causing memory issues. No important information is lost in the process.
    """


print(truncate("Ferdous ฿hai\n>>>gm"))
