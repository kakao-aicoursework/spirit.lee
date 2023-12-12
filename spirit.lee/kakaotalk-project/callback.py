# -*- coding: utf-8 -*-

from dto import ChatbotRequest
from samples import simple_text_sample
from openai import OpenAI
import aiohttp
import time
import os
from dotenv import load_dotenv
import requests
import json
from icecream import ic

load_dotenv()

SYSTEM_MSG = "당신은 카카오 서비스 제공자입니다."

async def callback_handler(request: ChatbotRequest) -> dict:
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": request.userRequest.utterance},
            {"role": "assistant", "content": SYSTEM_MSG},
        ]
    )
    output_text = completion.choices[0].message.content

    time.sleep(1.0)

    url = request.userRequest.callbackUrl

    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": output_text
                    }
                }
            ]
        }
    }

    if url:
        async with aiohttp.ClientSession() as session:
            async with session.post(url=url, json=payload) as resp:
                await resp.json()

# def callback_handler2(request: ChatbotRequest):
#     url = request.userRequest.callbackUrl
#     ic(url)
#     payload = {
#         "version": "2.0",
#         "template": {
#             "outputs": [
#                 {
#                     "simpleText": {
#                         "text": "콜백 응답~"
#                     }
#                 }
#             ]
#         }
#     }

#     try:
#         headers = { 'Content-Type': 'application/json' }
#         res = requests.post(url, headers=headers, data=json.dumps(payload))
#         ic(res.text)
#     except Exception as e:
#         ic(e)
