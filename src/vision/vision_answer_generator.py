from dotenv import load_dotenv

from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage


load_dotenv()


class VisionAnswerGenerator:

    MODEL_NAME = (
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    )

    def __init__(self):

        self.llm = ChatOpenRouter(
            model=self.MODEL_NAME,
            temperature=0,
        )

    def answer_from_image(
        self,
        question,
        image_path,
        caption,
    ):
        """
        Ask the vision model to analyze the image.
        """

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        f"Question: {question}\n\n"
                        f"Caption: {caption}\n\n"
                        "Answer the question using the image."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": image_path,
                },
            ]
        )

        response = self.llm.invoke([message])

        return response.content