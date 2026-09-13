from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage

import base64

load_dotenv()

with open(
    "data/images/Denseresultstable_p8_fig7.png",
    "rb",
) as file:
    image_base64 = base64.b64encode(
        file.read()
    ).decode()

llm = ChatOpenRouter(
    model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    temperature=0,
)

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "What does this figure show?"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": (
                    f"data:image/png;base64,"
                    f"{image_base64}"
                )
            }
        }
    ]
)

response = llm.invoke([message])

print(response.content)