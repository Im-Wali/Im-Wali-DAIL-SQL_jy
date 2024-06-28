import json.decoder

import openai
from utils.enums import LLM
import time
import transformers
import torch

def init_chatgpt(OPENAI_API_KEY, OPENAI_GROUP_ID, model):
    # if model == LLM.TONG_YI_QIAN_WEN:
    #     import dashscope
    #     dashscope.api_key = OPENAI_API_KEY
    # else:
    #     openai.api_key = OPENAI_API_KEY
    #     openai.organization = OPENAI_GROUP_ID
    openai.api_key = OPENAI_API_KEY
    openai.organization = OPENAI_GROUP_ID


def ask_completion(model, batch, temperature):
    response = openai.Completion.create(
        model=model,
        prompt=batch,
        temperature=temperature,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=[";"]
    )
    response_clean = [_["text"] for _ in response["choices"]]
    return dict(
        response=response_clean,
        **response["usage"]
    )


def ask_chat(model, messages: list, temperature, n):
    # response = openai.ChatCompletion.create(
    #     model=model,
    #     messages=messages,
    #     temperature=temperature,
    #     max_tokens=200,
    #     n=n
    # )
    # response_clean = [choice["message"]["content"] for choice in response["choices"]]
    # if n == 1:
    #     response_clean = response_clean[0]
    # return dict(
    #     response=response_clean,
    #     **response["usage"]
    # )
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )

    # messages = [
    #     {"role": "user", "content": "Who are you?"},
    # ]

    prompt = pipeline.tokenizer.apply_chat_template(
            messages , 
            tokenize=False, 
            add_generation_prompt=True
    )

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    print(outputs[0]["generated_text"][len(prompt):])
    # return [{"response": {"message": {
    #                     "content": output["generated_text"][len(prompt):].strip()}} for output in outputs}]
    return {"response": {
                        "content": output["generated_text"][len(prompt):].strip()} for output in outputs} 
    


def ask_llm(model: str, batch: list, temperature: float, n:int):
    n_repeat = 0
    while True:
        try:
            if model in LLM.TASK_COMPLETIONS:
                # TODO: self-consistency in this mode
                assert n == 1
                response = ask_completion(model, batch, temperature)
            elif model in LLM.TASK_CHAT:
                # batch size must be 1
                assert len(batch) == 1, "batch must be 1 in this mode"
                messages = [{"role": "user", "content": batch[0]  + '\n  Please show only the SQL results.'}]
                response = ask_chat(model, messages, temperature, n)
                # response['response'] = [response['response']]
            break
        except openai.error.RateLimitError:
            n_repeat += 1
            print(f"Repeat for the {n_repeat} times for RateLimitError", end="\n")
            time.sleep(1)
            continue
        except json.decoder.JSONDecodeError:
            n_repeat += 1
            print(f"Repeat for the {n_repeat} times for JSONDecodeError", end="\n")
            time.sleep(1)
            continue
        except Exception as e:
            n_repeat += 1
            print(f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
            time.sleep(1)
            continue

    return response

