import logging
import sqlite3

import lamini
import pandas as pd

from dotenv import load_dotenv
from util.get_schema import get_schema
from util.make_llama_3_prompt import make_llama_3_prompt
from util.setup_logging import setup_logging


_ = load_dotenv()

## ------------------------------------------------------ ##
logger = logging.getLogger(__name__)
engine = sqlite3.connect("./nba_roster.db")
setup_logging()

## ------------------------------------------------------ ##
llm = lamini.Lamini(model_name = "meta-llama/Meta-Llama-3-8B-Instruct")

## ------------------------------------------------------ ##
def make_llama_3_prompt(user, system = ""):
    system_prompt = ""

    if system != "":
        system_prompt = (f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>")

    return (f"<|begin_of_text|>{system_prompt}"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user}"
            f"<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n")

################
# Or - version 1
# def make_llama_3_prompt(user, system = ""):
#     system_prompt = ""

#     if system != "":
#         system_prompt = (f"<|start_header_id|>system<|end_header_id|>\n\n{system}"
#                         f"<|eot_id|>")

#     return (f"<|begin_of_text|{system_prompt}<|start_header_id|>user<|end_header_id|>\n\n{user}"
#             f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
################
# Or - version 2
# def make_llama_3_prompt(user, system = ""):
#     parts = {'system' : system if system else "", 'user' : user, }

#     template = """
#                 <|begin_of_text|>
#                 <|start_header_id|>system<|end_header_id|>
#                 {system}
#                 <|eot_id|>
#                 <|start_header_id|>user<|end_header_id|>
#                 {user}
#                 <|eot_id|>
#                 <|start_header_id|>assistant<|end_header_id|>
#               """

#     return template.format_map(parts)
################

## ------------------------------------------------------ ##
def get_schema():
    return """\
            0|Team|TEXT
            1|NAME|TEXT
            2|Jersey|TEXT
            3|POS|TEXT
            4|AGE|INT
            5|HT|TEXT
            6|WT|TEXT
            7|COLLEGE|TEXT
            8|SALARY|TEXT eg.
            """

## ------------------------------------------------------ ##
user = """Who is the highest paid NBA player?"""

## ------------------------------------------------------ ##
system = f"""You are an NBA analyst with 15 years of experience writing complex SQL queries.
            Consider the nba_roster table with the following schema:
            {get_schema()}

            Write a sqlite query to answer the following question. Follow instructions exactly"""

## ------------------------------------------------------ ##
print(system)

## ------------------------------------------------------ ##
prompt = make_llama_3_prompt(user, system)

## ------------------------------------------------------ ##
print(llm.generate(prompt, max_new_tokens = 200))

## ------------------------------------------------------ ##
def get_updated_schema():
    return """\
            0|Team|TEXT eg. "Toronto Raptors"
            1|NAME|TEXT eg. "Otto Porter Jr."
            2|Jersey|TEXT eg. "0" and when null has a value "NA"
            3|POS|TEXT eg. "PF"
            4|AGE|INT eg. "22" in years
            5|HT|TEXT eg. `6' 7"` or `6' 10"`
            6|WT|TEXT eg. "232 lbs"
            7|COLLEGE|TEXT eg. "Michigan" and when null has a value "--"
            8|SALARY|TEXT eg. "$9,945,830" and when null has a value "--"
            """

## ------------------------------------------------------ ##
system = f"""You are an NBA analyst with 15 years of experience writing complex SQL queries.
            Consider the nba_roster table with the following schema:
            {get_updated_schema()}

            Write a sqlite query to answer the following question. Follow instructions exactly"""

## ------------------------------------------------------ ##
print(prompt)

## ------------------------------------------------------ ##
print(llm.generate(prompt, max_new_tokens = 200))

## ------------------------------------------------------ ##
result = llm.generate(prompt, output_type = {"sqlite_query" : "str"}, max_new_tokens = 200)

## ------------------------------------------------------ ##
print(result)

## ------------------------------------------------------ ##
df = pd.read_sql(result['sqlite_query'], con = engine)

## ------------------------------------------------------ ##
print(df)

## ------------------------------------------------------ ##
query = """ SELECT salary, name
            FROM nba_roster
            WHERE salary != '--'
            ORDER BY CAST(REPLACE(REPLACE(salary, '$', ''), ',','') AS INTEGER) DESC
            LIMIT 1; """

df = pd.read_sql(query, con = engine)
print(df)
