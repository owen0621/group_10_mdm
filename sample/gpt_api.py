import os
from openai import OpenAI
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in .env")

client = OpenAI(api_key=api_key)


class PromptSplitter:
    def __init__(self, model="gpt-4"):
        self.model = model
        self.system_prompt = (
            "You are a helpful assistant that splits a long human motion description "
            "into coherent motion segments. Each segment should correspond to a short, meaningful motion action. "
            "Use simple and concrete physical descriptions. Keep each part independent and grounded in physical movement.\n\n"
            "Here are examples of good splits:\n"
            "Example 1:\n"
            "Original: A man walks forward, waves at a friend, then jumps and cheers.\n"
            "Split:\n"
            "1. A man walks forward\n"
            "2. He waves at a friend\n"
            "3. He jumps and cheers\n\n"
            "Example 2:\n"
            "Original: A person squats low, then jumps to the left, lands and poses.\n"
            "Split:\n"
            "1. A person squats low\n"
            "2. The person jumps to the left\n"
            "3. They land and pose\n\n"
            "Example 3:\n"
            "Original: A person falls into a seated position and stands up, then kicks with their left foot.\n"
            "Split:\n"
            "1. A person falls into a seated position\n"
            "2. They stand up\n"
            "3. They kick with their left foot\n\n"
            "Now split the user-provided motion into clear steps."
        )

    def split_prompt(self, text_prompt: str, clip_num: int):
        user_prompt = (
            f"Split the following human motion description into {clip_num} coherent parts.\n"
            f"Each part should describe a distinct short motion (like walking, waving, jumping, squatting).\n"
            f"Use only plain text, one line per part. Do not include any numbering, headings, or explanations.\n\n"
            f"{text_prompt}"
        )

        try:
            chat_completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )

            reply = chat_completion.choices[0].message.content.strip()

            segments = [
                line.strip("- ").strip("0123456789. ").strip()
                for line in reply.split("\n")
                if line.strip()
            ]

            if len(segments) < clip_num:
                print(f"[PromptSplitter Warning] GPT returned only {len(segments)} segments, duplicating last to match {clip_num}")
                while len(segments) < clip_num:
                    segments.append(segments[-1])

            return segments[:clip_num]

        except Exception as e:
            print(f"[PromptSplitter Error] {e}")
            return [text_prompt] * clip_num


# Example test
if __name__ == "__main__":
    splitter = PromptSplitter()
    prompt = "A person enters the room, waves to a friend, jumps up, then starts dancing excitedly in the middle."
    clip_num = 4
    segments = splitter.split_prompt(prompt, clip_num)

    print("\n--- Split Result ---")
    for i, seg in enumerate(segments):
        print(f"[Clip {i+1}] {seg}")
