import os

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Response format model
class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Extract the event information"},
        {
            "role": "user",
            "content": "Alice and Bob are going to a science fair on Friday.",
        },
    ],
    response_format=CalendarEvent,
)

response = completion.choices[0].message.parsed
print(response)