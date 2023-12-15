from dotenv import load_dotenv
from langchain.chains.router.embedding_router import EmbeddingRouterChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain, LLMChain, LLMRouterChain, MultiPromptChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains import RetrievalQA
import os
import glob
from pathlib import Path
import vector_db

load_dotenv()

def read_prompt_from_file(filename: str) -> str:
    with open(filename, "r") as f:
        return f.read()

def init_prompt() -> list:
    prompt_infos = []
    for file_path in glob.glob("./prompt/*.txt"):
        file_name = Path(file_path).stem
        prompt_text = read_prompt_from_file(file_path)
        prompt_info = {
            "name": file_name,
            "description": "Good for answering questions associated with '{}' topics.".format(file_name),
            "prompt_template": prompt_text
        }
        prompt_infos.append(prompt_info)
    return prompt_infos



# https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed
def create_qa_chain(llm, prompt_info, docs, related_content_number: int):
    ## qa chain을 쓰지 말고, llm chain + 직접 db query 날려서 context에 추가하는 방법밖에 없는 듯.
    # qa chain을 쓰면 query라는 변수명이 강제되는데, multi prompt에서는 input으로밖에 변수명이 안 들어가는 듯하다.

    # qa_chain_prompt = PromptTemplate(
    #     template=prompt_info['prompt_template'],
    #     input_variables=['context','input'],
    # )
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm,
    #     retriever=docs.as_retriever(search_kwargs={'k': related_content_number}),
    #     return_source_documents=True,
    #     chain_type_kwargs={"prompt": qa_chain_prompt}
    # )
    return qa_chain


def set_router_chain(llm, prompt_infos: list):
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destination_str = "\n".join(destinations)
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destination_str) ## input key 강제하는 부분.
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=['input'],
        output_parser=RouterOutputParser()
    )

    # EmbeddingRouterChain.from_names_and_descriptions(
    #     destinations,
    # )
    router_chain = LLMRouterChain.from_llm(llm, router_prompt)
    return router_chain

def set_multi_prompt_chain(llm, prompt_infos: list, docs):
    table = dict()
    for idx, prompt_info in enumerate(prompt_infos):
        table[prompt_info['name']] = idx

    return MultiPromptChain(
        router_chain=set_router_chain(llm, prompt_infos),
        destination_chains= {
            "카카오소셜": create_qa_chain(llm, prompt_infos[table['카카오소셜']], docs['카카오소셜'], 3),
            "카카오싱크": create_qa_chain(llm, prompt_infos[table["카카오싱크"]], docs['카카오싱크'], 3),
            "카카오톡채널": create_qa_chain(llm,prompt_infos[table["카카오톡채널"]], docs['카카오톡채널'], 3)
        },
        default_chain=ConversationChain(llm=llm, output_key="text"),
        verbose=True,
    )


from langchain.chains.base import Chain

class InputAdapterChain(Chain):
    def __init__(self, destination_chain, input_key_map):
        self.destination_chain = destination_chain
        self.input_key_map = input_key_map

    def run(self, inputs):
        adapted_inputs = {self.input_key_map[k]: v for k, v in inputs.items()}
        return self.destination_chain.run(adapted_inputs)

# def set_destination_chain(llm, prompt_infos: list):
#     destination_chains = dict()
#     for p_info in prompt_infos:
#         name = p_info['name']
#         prompt_template = p_info['prompt_template']
#         prompt = PromptTemplate(template=prompt_template, input_variables=['user_message', 'related_documents'])
#         chain = LLMChain(llm=llm, prompt=prompt)
#         destination_chains[name] = chain
#     return destination_chains

# def set_default_chain(llm):
#     return ConversationChain(llm=llm, output_key='text')



# def set_multi_prompt_chain(llm, prompt_infos: list, docs):
    # return MultiPromptChain(
    #     router_chain=set_router_chain(llm, prompt_infos),
    #     destination_chain=set_destination_chain(llm, prompt_infos),
    #     default_chain=set_default_chain(llm),
    #     verbose=True
    # )