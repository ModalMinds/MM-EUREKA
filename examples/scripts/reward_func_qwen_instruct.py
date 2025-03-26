import os
import re
from datetime import datetime

import regex
import torch
from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify
from math_verify import parse, verify
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
LOG_PATH = os.environ.get("REWARD_LOG_PATH", "reward.log")
import re



choices = ['a','b','c','d']
problem_pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
response_prefix = r"<\|im_start\|>assistant\n"

def get_response_from_query(q: str):
    ends_of_sentence = ["<|im_end|>", "<｜end▁of▁sentence｜>", "<|endoftext|>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return ""
    response = q[pos.end() :]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()

def get_query_from_query(q: str):
    # 使用 re.findall 提取匹配的内容
    try:
        matches = re.findall(problem_pattern, q, re.DOTALL)
        return matches[0]
    except:
        return q




# q = """<|im_start|>system\nYou are a helpful assistant good at solving math problems with step-by-step reasoning. You should first thinks about the reasoning process in the mind and then provides the user with the answer. Your answer must be in latex format and wrapped in $...$.The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> $2$ </answer>, which means your output should start with <think> and end with </answer>.<|im_end|>\n<|im_start|>user\nConsider the function $g(x) = 3$.  Find $g(2)$.<|im_end|>\n<|im_start|>assistant\n<think> Let's start by understanding the function $g(x)$, which is defined as $g(x) = 3$. This means that no matter what value we substitute for $x$, the function will always return the constant value of 3. Therefore, when we want to find $g(2)$, we substitute $x = 2$ into the function. This gives us $g(2) = 3$.\n\n</think><answer> $3$ </answer><|im_end|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"""
# print(get_response_from_query(q))
# print(get_query_from_query(q))





def extract_answer_with_tags(text):
    # 使用正则表达式匹配包括 <answer> 和 </answer> 的内容
    match = re.search(r"(<answer>.*?</answer>)", text)
    if match:
        return match.group(1)  # 返回匹配的内容，包括标签
    return None  # 如果没有匹配到，返回 None


def accuracy_reward_func(completion, answer):
    reward = 0.0
    response = extract_answer_with_tags(completion)
    # print(f"匹配到的内容: {response}")
    if response != None:

        # print(accuracy_reward_func(response,gt))
        response = response

    else:
        try:
            response = completion.split('<answer>')[-1]
        except:
            response = completion.split('\n')[-1]
        
        
    content, sol = response, answer
    answer_parsed = content
    sol = f"${str(sol)}$"
    gold_parsed = parse(sol)
    if len(gold_parsed) != 0:
        # We require the answer to be provided in correct latex (no malformed operators)
        answer_parsed = parse(
                content,
                extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()],
            )
        # Reward 1 if the content is the same as the ground truth, 0 otherwise
        # print(answer_parsed, gold_parsed)
        try:
            reward = float(verify(answer_parsed, gold_parsed))
        except Exception as e:
            pass
        
        if reward == 0.0:
            try:
                content_match = re.search(r'<answer>(.*?)</answer>', completion)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                student_answer = student_answer.replace('</answer>','').replace('<answer>','').strip()
                for answer in gold_parsed:
                    if str(answer).lower() in choices:
                         # 说明是选择题
                        if str(answer).lower() in student_answer.lower():
                            # print("str(answer) in student_answer.lower():")
                            
                            choices_other = [choice for choice in choices if choice != str(answer).lower()]
                            # print('choices_other:' ,choices_other)
                            # print('student_answer: ',student_answer.lower())
                            if all(choice not in student_answer.lower() for choice in choices_other):
                                 reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
    else:
        # If the gold solution is not parseable, we reward 1 to skip this example
        reward = 1.0
        print("Failed to parse gold solution: ", sol)
        


    return reward, answer_parsed
        



# def format_reward_func(completion):
#     pattern = (
#         r"^(?=(?:.*<think>){1})(?=(?:.*<\/think>){1})"
#         r"(?!.*<think>.*<think>)"
#         r"(?!.*<\/think>.*<\/think>)"
#         r".*<think>.*?</think>.*$"
#     )
#     matches = re.search(pattern, completion, re.DOTALL)
#     return 1.0 if matches else 0.0


def format_reward_func(completion, **kwargs):
    """Reward function that allows some content between </think> and <answer>,
    and also allows content before <think>.
    """
    # pattern = r"<think>.*?</think>.*?<answer>.*?</answer>"
    pattern = (
        r"^(?=(?:.*<think>){1})(?=(?:.*<\/think>){1})"
        r"(?=(?:.*<answer>){1})(?=(?:.*<\/answer>){1})" 
        r"(?!.*<think>.*<think>)"
        r"(?!.*<\/think>.*<\/think>)"
        r"(?!.*<answer>.*<answer>)"
        r"(?!.*<\/answer>.*<\/answer>)"
        r".*<think>(.+?)</think>\s*<answer>.+?</answer>.*$"
    )
    matches = re.search(pattern, completion, re.DOTALL)
    return 0.5 if matches else 0.0


def reward_func(queries, prompts, labels):
    # queries is prompts + responses

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []
    accuracy_rewards = []
    format_rewards = []
    # pattern = r"<\|im_start\|> assistant(.*?)<\|im_end\|>"
    # pattern = r"<\|im_start\|>\s*assistant(.*?)<\|im_end\|>"

    # print("reward_func_queries: ", queries)
    # print("reward_func_prompts: ", prompts)
    with open(LOG_PATH, "a") as f:
        f.write(f"----------------------------- {current_time} -----------------------------\n")
        for query, prompt, answer in zip(queries, prompts, labels):
            # response = query
            # print("reward_func: ", prompt)
            # try:
                # query = re.sub(r"\s*<IMG_CONTEXT>\s*", "", query)
                # query = re.sub(r"<img>\s*</img>", " <image>", query)
                # query = re.sub("</s>", "", query)
                # response = re.search(pattern, query, re.DOTALL).group(1).strip()
            try:
                response = get_response_from_query(query)
                if response == "":
                    f.write("Error: " + query + "\n")
                    rewards.append(0.0)
                    accuracy_rewards.append(0.0)
                    format_rewards.append(0.0)
                
                else:
                    query1 = get_query_from_query(query)
                    # answer = prompt["answer"]
                    answer = answer

                    accuracy_reward, answer_parsed = accuracy_reward_func(response, answer)
                    format_reward = format_reward_func(response)
                    
                    

                    rewards.append(accuracy_reward + format_reward)
                    accuracy_rewards.append(accuracy_reward)
                    format_rewards.append(format_reward)
                    f.write(f"===============================================================\n")
                    f.write("Query: " + query1 + "\n")
                    f.write("Response: " + response + "\n")
                    f.write("Answer: " + answer + "\n")
                    f.write(f"Accuracy Reward: {accuracy_reward}\tFormat Reward: {format_reward}\n\n\n\n")
                    f.write(f"===============================================================\n")
            except:
                f.write("Error: " + query + "\n")
                rewards.append(0.0)
                accuracy_rewards.append(0.0)
                format_rewards.append(0.0)

    return {
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "accuracy_rewards": torch.tensor(accuracy_rewards, dtype=torch.float32),
        "format_rewards": torch.tensor(format_rewards, dtype=torch.float32),
    }
