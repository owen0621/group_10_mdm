import os
import re
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

        # 系統提示：用於只拆分 text prompt（不附秒數）的模式
        self.system_prompt_text_only = (
            "You are a helpful assistant that splits a long human motion description "
            "into coherent motion segments. Each segment should correspond to a short, meaningful motion action. "
            "Use simple and concrete physical descriptions. Keep each part independent and grounded in physical movement.\n\n"
            "Examples:\n"
            "Original: A man walks forward, waves at a friend, then jumps and cheers.\n"
            "Split:\n"
            "1. A man walks forward\n"
            "2. He waves at a friend\n"
            "3. He jumps and cheers\n\n"
            "Original: A person squats low, then jumps to the left, lands and poses.\n"
            "Split:\n"
            "1. A person squats low\n"
            "2. The person jumps to the left\n"
            "3. They land and pose\n\n"
            "Now split the user-provided motion into exactly the requested number of parts without numbering or explanations."
        )

        # 系統提示：用於拆分 text prompt 並附帶秒數的模式
        self.system_prompt_with_dur = (
            "You are a helpful assistant that splits a long human motion description "
            "into coherent motion segments. Each segment should correspond to a short, meaningful motion action. "
            "Use simple and concrete physical descriptions. Keep each part independent and grounded in physical movement.\n\n"
            "For each segment, also assign a playback duration (in seconds) between 3 and 8. "
            "Format each line exactly as: <motion description> - <duration>\n\n"
            "Examples:\n"
            "Original: A man walks forward, waves at a friend, then jumps and cheers.\n"
            "Split:\n"
            "A man walks forward - 4\n"
            "He waves at a friend - 3\n"
            "He jumps and cheers - 5\n\n"
            "Original: A person squats low, then jumps to the left, lands and poses.\n"
            "Split:\n"
            "A person squats low - 3\n"
            "They jump to the left - 4\n"
            "They land and pose - 3\n\n"
            "Original: A person falls into a seated position and stands up, then kicks with their left foot.\n"
            "Split:\n"
            "A person falls into a seated position - 3\n"
            "They stand up - 4\n"
            "They kick with their left foot - 3\n\n"
            "Now split the user-provided motion into exactly the requested number of parts, "
            "each with its own duration between 3 and 8 seconds."
        )

    def split_prompt(self, text_prompt: str, clip_num: int, with_duration: bool = False):
        """
        If with_duration=False, return a list of clip_num text-only segments.
        If with_duration=True, return a list of (segment_text, duration) tuples.
        """
        if not with_duration:
            # 只拆分文字，不附秒數
            user_prompt = (
                f"Split the following human motion description into {clip_num} coherent parts. "
                f"Each part should describe a distinct short motion (like walking, waving, jumping, squatting). "
                f"Output exactly {clip_num} lines of plain text, one segment per line, without numbering.\n\n"
                f"{text_prompt}"
            )
            try:
                chat_completion = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt_text_only},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7
                )
                reply = chat_completion.choices[0].message.content.strip()
                lines = [line.strip() for line in reply.split("\n") if line.strip()]

                segments = []
                for line in lines:
                    # 移除任何開頭編號或破折號
                    cleaned = re.sub(r"^[\-\d\.]+\s*", "", line).strip()
                    segments.append(cleaned)
                    if len(segments) >= clip_num:
                        break

                while len(segments) < clip_num:
                    segments.append(segments[-1])

                return segments[:clip_num]

            except Exception as e:
                print(f"[PromptSplitter Error] {e}")
                return [text_prompt] * clip_num

        else:
            # 拆分文字並附秒數
            user_prompt = (
                f"Split the following human motion description into {clip_num} coherent parts. "
                f"Each part should describe a distinct short motion (like walking, waving, jumping, squatting). "
                f"For each part, also assign an integer duration (in seconds) between 3 and 8. "
                f"Output exactly {clip_num} lines, formatted as '<motion description> - <duration>'.\n\n"
                f"{text_prompt}"
            )
            try:
                chat_completion = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt_with_dur},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7
                )
                reply = chat_completion.choices[0].message.content.strip()
                lines = [line.strip() for line in reply.split("\n") if line.strip()]

                segments_with_dur = []
                for line in lines:
                    # 嘗試解析 "<text> - <duration>"
                    match = re.match(r"^(.*?)\s*[-–]\s*(\d+)\s*$", line)
                    if match:
                        text = match.group(1).strip()
                        duration = int(match.group(2))
                        duration = max(3, min(duration, 8))
                        segments_with_dur.append((text, duration))
                    else:
                        # 若格式不符，整行當 text，給預設 5 秒
                        segments_with_dur.append((line, 5))

                    if len(segments_with_dur) >= clip_num:
                        break

                while len(segments_with_dur) < clip_num:
                    last_text, last_dur = segments_with_dur[-1]
                    segments_with_dur.append((last_text, last_dur))

                return segments_with_dur[:clip_num]

            except Exception as e:
                print(f"[PromptSplitter Error] {e}")
                return [(text_prompt, 5)] * clip_num


# Example test
if __name__ == "__main__":
    splitter = PromptSplitter()
    prompt = "A person enters the room, waves to a friend, jumps up, then starts dancing excitedly in the middle."
    clip_num = 4

    # 測試文字-only 模式
    segments = splitter.split_prompt(prompt, clip_num, with_duration=False)
    print("\n--- Text-only Split ---")
    for i, seg in enumerate(segments):
        print(f"[Clip {i+1}] {seg}")

    # 測試附秒數模式
    segments_dur = splitter.split_prompt(prompt, clip_num, with_duration=True)
    print("\n--- Split with Durations ---")
    for i, (seg, dur) in enumerate(segments_dur):
        print(f"[Clip {i+1}] {seg}   ({dur} sec)")
