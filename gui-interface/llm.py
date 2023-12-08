from langchain.chains.question_answering import load_qa_chain ## 자주 쓰는 걸 아예 함수로 빼놓은 것.
from langchain.llms import OpenAI
# from openai import OpenAI as OpenAIAPI
from langchain.prompts import PromptTemplate
import os

# def query(text: str, docs):
#     # openai 1.0.0부터는 아래 방식을 사용해야 한다고 함
#     # https://github.com/openai/openai-python
#
#     client = OpenAIAPI(
#         # This is the default and can be omitted
#         api_key=os.environ.get("OPENAI_API_KEY"),
#     )
#
#     completion = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "user", "content": "로맨틱 코미디 드라마 10개만 추천해줄래? 소개할때 드라마 제목, 장르, 출연진도 같이 소개해줘. 응답은 한글로 해줘"},
#             {"role": "assistant", "content": f"{docs}"},
#         ]
#     )
#     return completion



def query_by_langchain(docs, query):

    prompt_template = """
        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer in Korean:
    """
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=['context','question']
    )
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff", prompt=PROMPT, verbose=True)
    result = chain.run(input_documents=docs, question=query)

    return result
    # return chain.run(input_documents=docs, question=query, return_only_outputs=True)



