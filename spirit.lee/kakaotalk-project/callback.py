# -*- coding: utf-8 -*-
import langchain.chains.qa_with_sources

from dto import ChatbotRequest
from samples import simple_text_sample
from langchain.chat_models import ChatOpenAI
import llm
import time
import os
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate
import vector_db
import requests
import json
from icecream import ic

load_dotenv()

def query_by_langchain(docs, query) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    prompt_template = """
        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {summaries}

        Question: {question}
        Answer in Korean:
    """
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=['summaries','question']
    )
    # summaries 변수가 반드시 있어야 하는 형태의 함수. (에러로그 보고 알 수 있음. 공식문서나 api docs에 전혀 안 써있다.)
    chain = load_qa_with_sources_chain(llm, chain_type="stuff", prompt=PROMPT, verbose=True)
    # chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT, verbose=True)
    result = chain.run(input_documents=docs, question=query)

    return result
    # return chain.run(input_documents=docs, question=query, return_only_outputs=True)

def query_multi_prompt_langchain(docs, query):
    llm_model = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    prompt_infos = llm.init_prompt()
    ## intent classification
    ### Q. 이 정도 small task에서 굳이 intent 분류에 api 한 번, 해당 분류에 따른 작업처리로 api 한 번 호출해야 할 필요가 있을까
    parse_intent_chain = llm.create_chain(llm_model, prompt_infos['intent']['prompt_template'], 'intent')



    intent = parse_intent_chain.run({
        "context": "\n".join([f"{v['name']}: {v['description']}" if k != 'intent' else "" for k, v in prompt_infos.items()]),
        "input": query
    })
    if intent not in ("카카오소셜", "카카오톡채널", "카카오톡싱크"):
        result = {"answer": "not classified"}
    else:
        next_chain = llm.create_chain(llm_model, prompt_infos[intent]['prompt_template'], output_key="answer")
        context = vector_db.get_relevant_documents(docs[intent], 3, query)
        answer = next_chain.run({
            "context": context,
            "input": query
        })
        result = {"answer": answer}

    # chain = llm.set_multi_prompt_chain(llm_model, prompt_infos, docs)
    # result = chain.run(query)
    # print(result)
    return result


def callback_handler(request: ChatbotRequest, docs) -> dict:
    # client = OpenAI(
    #     # This is the default and can be omitted
    #     api_key=os.environ.get("OPENAI_API_KEY"),
    # )

    print(request)
    query = request.userRequest.utterance
    # related_docs = vector_db.get_relevant_documents(docs, 3, query)
    # output_text = query_by_langchain(related_docs, query)
    output_text = query_multi_prompt_langchain(docs, query)

    print(output_text['answer'])

    url = request.userRequest.callbackUrl

    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": output_text['answer']
                    }
                }
            ]
        }
    }

    try:
        headers = { 'Content-Type': 'application/json' }
        res = requests.post(url, headers=headers, data=json.dumps(payload))
        ic(res.text)
    except Exception as e:
        ic(e)
