import lamini

from dotenv import load_dotenv


_ = load_dotenv()

## ------------------------------------------------------ ##
llm = lamini.Lamini(model_name = "meta-llama/Meta-Llama-3-8B-Instruct")

## ------------------------------------------------------ ##
prompt = """\
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

        Please write a birthday card for my good friend Andrew\
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>

        """

## ------------------------------------------------------ ##
result = llm.generate(prompt, max_new_tokens = 200)
print(result)

## ------------------------------------------------------ ##
prompt2 = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant."
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Please write a birthday card for my good friend Andrew"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )

print(prompt2)

## ------------------------------------------------------ ##
prompt == prompt2

## ------------------------------------------------------ ##
def make_llama_3_prompt(user, system = ""):
    system_prompt = ""

    if system != "":
        system_prompt = (
                        f"<|start_header_id|>system<|end_header_id|>\n\n{system}"
                        f"<|eot_id|>"
                        )

    prompt = (f"<|begin_of_text|>{system_prompt}"
              f"<|start_header_id|>user<|end_header_id|>\n\n"
              f"{user}"
              f"<|eot_id|>"
              f"<|start_header_id|>assistant<|end_header_id|>\n\n"
             )

    return prompt

## ------------------------------------------------------ ##
system_prompt = user_prompt = "You are a helpful assistant."
user_prompt = "Please write a birthday card for my good friend Andrew"
prompt3 = make_llama_3_prompt(user_prompt, system_prompt)
print(prompt3)

## ------------------------------------------------------ ##
prompt == prompt3

## ------------------------------------------------------ ##
user_prompt = "Tell me a joke about birthday cake"
prompt = make_llama_3_prompt(user_prompt)
print(prompt)

## ------------------------------------------------------ ##
result = llm.generate(prompt, max_new_tokens = 200)
print(result)

## ------------------------------------------------------ ##
user_prompt = "Tell me a joke about American indian and cowboy"
prompt = make_llama_3_prompt(user_prompt)
result = llm.generate(prompt, max_new_tokens = 200)
print(result)

## ------------------------------------------------------ ##
question = (
            "Given an arbitrary table named `sql_table`, "
            "write a query to return how many rows are in the table."
            )

prompt = make_llama_3_prompt(question)
print(llm.generate(prompt, max_new_tokens = 200))

## ------------------------------------------------------ ##
question = """Given an arbitrary table named `sql_table`,
            help me calculate the average `height` where `age` is above 20.
            """

prompt = make_llama_3_prompt(question)
print(llm.generate(prompt, max_new_tokens = 200))

## ------------------------------------------------------ ##
question = """Given an arbitrary table named `sql_table`,
            Can you calculate the p95 `height` where the `age` is above 20?
            """

prompt = make_llama_3_prompt(question)
print(llm.generate(prompt, max_new_tokens = 200))

## ------------------------------------------------------ ##
question = ("Given an arbitrary table named `sql_table`, "
            "Can you calculate the p95 `height` "
            "where the `age` is above 20? Use sqlite.")

prompt = make_llama_3_prompt(question)
print(llm.generate(prompt, max_new_tokens = 200))

## ------------------------------------------------------ ##
question = ("Given this SQL code: "
            '''
            SELECT NTILE(100, height) AS p95_height
            FROM (
              SELECT height
              FROM sql_table
              WHERE age > 20
            ) AS subquery
            ORDER
            ,'''
            "Can you explain it to me?")

prompt = make_llama_3_prompt(question)
print(llm.generate(prompt, max_new_tokens = 200))

## ------------------------------------------------------ ##


## ------------------------------------------------------ ##


## ------------------------------------------------------ ##


## ------------------------------------------------------ ##