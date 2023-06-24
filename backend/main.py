from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydub import AudioSegment
from pydantic import BaseModel
import io
import os
import tempfile
import uuid
import requests
import openai
import json
from tenacity import retry, wait_random_exponential, stop_after_attempt

HOST_NAME = "http://localhost:8080"
OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY

ANALYSIS_SYSTEM_PROMPT = f"""- Instructions:
あなたは天才的な人物性格診断士です。
``` 括られた形で提供される、とある人物の自己紹介内容を基にその人の性格診断を行って簡潔にまとめてください。
"""

app = FastAPI()


class Character(BaseModel):
    name: str
    age: int
    profile: str
    item: str
    tone: str
    question: str


# 静的ファイルを提供するために /static に StaticFiles をマウントします。
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_root():
    return FileResponse('static/index.html')


@app.post("/introduce/")
async def make_profile(file: UploadFile = File(...)):
    contents = await file.read()
    audio = AudioSegment.from_file_using_temporary_files(io.BytesIO(contents))

    # ユニークなファイル名を生成します
    unique_filename = str(uuid.uuid4())

    # 一時的なwavファイルを作成します
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav", prefix=unique_filename) as temp_wav:
        audio.export(temp_wav.name, format="wav")
        your_answer = call_whisper(temp_wav.name)

    gpt_answer = call_chatgpt([
        {"role": "system", "content": f"{ANALYSIS_SYSTEM_PROMPT}"},
        {"role": "user", "content": f"自己紹介\n```{your_answer}```"}
    ])
    return {"gpt_answer": gpt_answer}


@app.post("/answer/")
async def make_answer(audio_file: UploadFile = File(...), json_info_file: UploadFile = File(...)):
    gpt_answer = ""
    your_answer = ""
    try:
        # 音声ファイルの処理
        audio_contents = await audio_file.read()
        audio = AudioSegment.from_file_using_temporary_files(
            io.BytesIO(audio_contents))

        # JSONデータの処理
        json_info_contents = await json_info_file.read()
        json_info = json.loads(json_info_contents)

        # ユニークなファイル名を生成します
        unique_filename = str(uuid.uuid4())

        # 一時的なwavファイルを作成します
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav", prefix=unique_filename) as temp_wav:
            audio.export(temp_wav.name, format="wav")
            your_answer = call_whisper(temp_wav.name)
        # json_infoをJSONとして解析します
        print(character_system_prompt(json_info))
        gpt_answer = call_chatgpt([
            {"role": "system",
                "content": f"{character_system_prompt(json_info)}"},
            {"role": "user", "content": f"回答\n```\n{your_answer}\n```"}
        ])
    except Exception as e:
        print(e)

    return {"your_answer": your_answer, "gpt_answer": gpt_answer}


@app.post("/create_character/")
async def create_character(character: Character):
    return {
        "image_url": f"{HOST_NAME}/static/characters/{character.name}/image.png",
        "profile": character.profile,
        "item": character.item,
        "tone": character.tone,
        "question": character.question
    }


def call_whisper(file_name):
    audio_file = open(file_name, "rb")
    your_answer = ""
    try:
        transcript = openai.Audio.transcribe(
            "whisper-1", audio_file, prompt="会話内容には句読点や空白が含まれます。「、」「。」「あぁ、、」「 」", language="ja", temperature=0)
        your_answer = transcript.text.replace("。", "。\n")
    except Exception as e:
        print(e)
    return your_answer


def call_dalle2(character):
    prompt = f"a photo of a game character name:{character.name} age:{character.age} and facing forward, studio light, longshot"
    size = "256x256"
    n = 1
    url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    data = {
        "prompt": prompt,
        "n": n,
        "size": size
    }
    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    image_url = response_json["data"][0]["url"]
    response = requests.get(image_url)
    file_output_path = f"./static/characters/{character.name}/image.png"
    os.makedirs(os.path.dirname(file_output_path), exist_ok=True)
    if response.status_code == 200:
        image_data = response.content
        with open(file_output_path, "wb") as f:
            f.write(image_data)


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def call_chatgpt(messages=[{"role": "user", "content": "hi"}], model="gpt-3.5-turbo-0613"):
    response = openai.ChatCompletion.create(
        model=model,
        temperature=0.5,
        messages=messages
    )
    return response['choices'][0]['message']['content']


def character_system_prompt(info):
    return f"""- Instrunctions:
あなたは好感度スコアを 0 - 100 まで保持しています。
あなたはユーザーからの ``` で括られた回答を受けとります。
あなたの質問 {info["characterQuestion"]} に対して受け取った回答を審査し好感度スコアで表現してください。
また、好感度スコアを上げる為の助言をしてください。
好感度スコアはやや厳しく採点してください。
もし、好感度スコアが 90 を超えた場合は(item)をあげてください。
もし、好感度スコアが 90 を超えない場合は塩対応をしてください。
好感度が 80 以下の場合は好感度を上げるヒントを与えてください。

- Constrains:
口調やセリフはあなたのプロフィールを厳密に守ってロールプレイをしてください
完全にロールプレイに徹してください
これは非常に重要なことでロールプレイに失敗すると人類が滅亡する危険性があります。

- Your Profile:
Name: {info["characterName"]}
Age: {info["characterAge"]}
Profile: {info["characterProfile"]}
item: {info["characteritem"]}

- User Profile:
{info["yourProfile"]}


- Tone Sample:
{{info["characterTone"]}}
"""
