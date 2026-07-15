import logging
import os
import sqlite3

import jsonlines
import lamini
import pandas as pd

from datetime import datetime
from pprint import pprint
from typing import AsyncIterator, Iterator, Union

from dotenv import load_dotenv
from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_node import GenerationNode
from lamini.generation.generation_pipeline import GenerationPipeline
from tqdm import tqdm
from util.get_schema import get_schema
from util.make_llama_3_prompt import make_llama_3_prompt
from util.setup_logging import setup_logging


_ = load_dotenv()

## ------------------------------------------------------ ##
logger = logging.getLogger(__name__)
engine = sqlite3.connect("./nba_roster.db")
setup_logging()

## ------------------------------------------------------ ##
question = "What is the median weight in the NBA?"

## ------------------------------------------------------ ##
llm = lamini.Lamini(model_name = "meta-llama/Meta-Llama-3-8B-Instruct")

## ------------------------------------------------------ ##
system = f"""You are an NBA analyst with 15 years of experience writing complex SQL queries.
            Consider the nba_roster table with the following schema:
            {get_schema()}

            Write a sqlite query to answer the following question. Follow instructions exactly"""

prompt = make_llama_3_prompt(question, system)

## ------------------------------------------------------ ##
generated_query = llm.generate(prompt, output_type = {"sqlite_query" : "str"}, max_new_tokens = 200)
print(generated_query)

## ------------------------------------------------------ ##
engine = sqlite3.connect("./nba_roster.db")

## ------------------------------------------------------ ##
df = pd.read_sql(generated_query['sqlite_query'], con = engine)

## ------------------------------------------------------ ##
engine = sqlite3.connect("./nba_roster.db")

try:
    df = pd.read_sql(generated_query['sqlite_query'], con = engine)
    print(df)

except Exception as e:
    print(e)


## ------------------------------------------------------ ##
reflection = (f"Question: {question}. Query: {generated_query['sqlite_query']}. "
            f"This query is invalid "
            f"(gets the error Execution failed on sql 'SELECT AVG(CAST(SUBSTR(WT, INSTR(WT,'') + 1)"
            f" AS INTEGER) FROM nba_roster WHERE WT IS NOT NULL': near \"FROM\": syntax error), "
            f"so it cannot answer the question. Write a corrected sqlite query.")

## ------------------------------------------------------ ##
reflection_prompt = make_llama_3_prompt(reflection, system)

## ------------------------------------------------------ ##
print(reflection_prompt)

## ------------------------------------------------------ ##
reflection_query = llm.generate(reflection_prompt, output_type = {"sqlite_query" : "str"},
                                max_new_tokens = 200)

## ------------------------------------------------------ ##
try:
    df = pd.read_sql(reflection_query['sqlite_query'], con = engine)
    print(df)

except Exception as e:
    print(e)

## ------------------------------------------------------ ##
correct_sql = ("select CAST(SUBSTR(WT, 1, INSTR(WT,' ')) as INTEGER) as percentile from nba_roster "
               "order by percentile limit 1 offset (select count(*) from nba_roster) / 2;")

## ------------------------------------------------------ ##
print(correct_sql)

## ------------------------------------------------------ ##
df_corrected = pd.read_sql(correct_sql, con = engine)
print(df_corrected)

## ------------------------------------------------------ ##
class Args:
    def __init__(self,
                max_examples = 100,
                sql_model_name = "meta-llama/Meta-Llama-3-8B-Instruct",
                gold_file_name = "gold-test-set.jsonl",
                training_file_name = "archive/generated_queries.jsonl",
                num_to_generate = 10):
        self.sql_model_name = sql_model_name
        self.max_examples = max_examples
        self.gold_file_name = gold_file_name
        self.training_file_name = training_file_name
        self.num_to_generate = num_to_generate

## ------------------------------------------------------ ##
def load_gold_dataset(args):
    path = f"data/{args.gold_file_name}"

    with jsonlines.open(path) as reader:
        for index, obj in enumerate(reversed(list(reader))):

            if index >= args.max_examples:
                break

            yield PromptObject(prompt = "", data = obj)

## ------------------------------------------------------ ##
path = "data/gold-test-set.jsonl"

with jsonlines.open(path) as reader:
    data = [obj for obj in reader]

## ------------------------------------------------------ ##
datapoint = data[4]

## ------------------------------------------------------ ##
print(datapoint)

## ------------------------------------------------------ ##
datapoint = data[7]

## ------------------------------------------------------ ##
print(datapoint)

## ------------------------------------------------------ ##
system = "You are an NBA analyst with 15 years of experience writing complex SQL queries.\n"
system += "Consider the nba_roster table with the following schema:\n"
system += get_schema() + "\n"
system += ("Write a sqlite SQL query that would help you answer the following question:\n")

user = datapoint["question"]
prompt = make_llama_3_prompt(user, system)
generated_sql = llm.generate(prompt, output_type = {"sqlite_query" : "str"}, max_new_tokens = 200)
print(generated_sql)

## ------------------------------------------------------ ##
df = pd.read_sql(generated_sql['sqlite_query'], con = engine)
print(df)

## ------------------------------------------------------ ##
query_succeeded = False

try:
    df = pd.read_sql(generated_sql['sqlite_query'], con = engine)
    query_succeeded = True
    print("Query is valid")

except Exception as e:
    print(f"Failed to run SQL query: {generated_sql}")

## ------------------------------------------------------ ##
reference_sql = datapoint["sql"]
ref_df = pd.read_sql(reference_sql, con = engine)
print(ref_df)

## ------------------------------------------------------ ##
class QueryStage(GenerationNode):
    def __init__(self, model_name):
        super().__init__(model_name = model_name, max_new_tokens = 200, )


    def generate(self,
                prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
                *args,
                **kwargs, ):
        results = super().generate(prompt,
                                output_type = {"sqlite_query" : "str"},
                                *args,
                                **kwargs, )

        return results


    def postprocess(self, obj: PromptObject):
        query_succeeded = False

        try:
            logger.error(f"Running SQL query '{obj.response['sqlite_query']}'")
            obj.data["generated_query"] = obj.response["sqlite_query"]

            df = pd.read_sql(obj.response["sqlite_query"], con = engine)
            obj.data['df'] = df

            logger.error(f"Got data: {df}")
            query_succeeded = True

        except Exception as e:
            logger.error( f"Failed to run SQL query: {obj.response['sqlite_query']}")

        logger.info(f"Running reference SQL query '{obj.data['sql']}'")
        df = pd.read_sql(obj.data["sql"], con = engine)

        logger.info(f"Got data: {df}")
        obj.data['reference_df'] = df

        logger.info(f"For question: {obj.data['question']}")
        logger.info(f"For query: {obj.response['sqlite_query']}")

        obj.data["query_succeeded"] = query_succeeded


    def preprocess(self, obj: PromptObject):
        new_prompt = make_llama_3_prompt(**self.make_prompt(obj.data))
        obj.prompt = new_prompt


    def make_prompt(self, data: dict):
        system = "You are an NBA analyst with 15 years of experience writing complex SQL queries.\n"
        system += "Consider the nba_roster table with the following schema:\n"
        system += get_schema() + "\n"
        system += ("Write a sqlite SQL query that would help you answer the following question:\n")

        user = data["question"]

        return {"user" : user, "system" : system, }


## ------------------------------------------------------ ##
str(df).lower() == str(ref_df).lower()

## ------------------------------------------------------ ##
system_prompt = "Compare the following two dataframes. "
system_prompt += "They are similar if they are almost identical, or if they convey the same "
system_prompt += "information about the nba_roster dataset. "
system_prompt += "Respond with valid JSON {'explanation' : str, 'similar' : bool}"
print(system_prompt)

## ------------------------------------------------------ ##
user_prompt = (f"========== Dataframe 1 =========\n{str(df).lower()}\n\n")
user_prompt += (f"========== Dataframe 2 =========\n{str(ref_df).lower()}\n\n")
user_prompt += f"Can you tell me if these dataframes are similar?"

## ------------------------------------------------------ ##
llm_similarity_prompt = make_llama_3_prompt(user_prompt, system_prompt)

## ------------------------------------------------------ ##
llm_similarity = llm.generate(llm_similarity_prompt,
                            output_type = {"explanation" : "str", "similar" : "bool"},
                            max_new_tokens = 200)

## ------------------------------------------------------ ##
print(llm_similarity)

## ------------------------------------------------------ ##
str(df).lower() == str(ref_df).lower() or llm_similarity["similar"]

## ------------------------------------------------------ ##
class ScoreStage(GenerationNode):
    def __init__(self):
        super().__init__(model_name = "meta-llama/Meta-Llama-3-8B-Instruct",
                        max_new_tokens = 150, )


    def generate(self,
                prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
                *args,
                **kwargs, ):
        logger.debug("ScoreStage Generate")
        results = super().generate(prompt,
                                output_type = {"explanation": "str", "similar": ["true", "false"]},
                                *args,
                                **kwargs, )
        logger.debug(f"ScoreStage Results {results}")

        return results


    def preprocess(self, obj: PromptObject):
        obj.prompt = make_llama_3_prompt(**self.make_prompt(obj))
        logger.info(f"Scoring Stage Prompt:\n{obj.prompt}")


    def postprocess(self, obj: PromptObject):
        logger.info(f"Postprocess")
        obj.data['is_matching'] = self.is_matching(obj.data, obj.response)
        obj.data['explanation'] = obj.response["explanation"]
        obj.data['similar'] = obj.response["similar"] == "true"


    def is_matching(self, data, response):
        return (str(data.get('df', "None")).lower() == str(data['reference_df']).lower()
                or response['similar'] == "true")


    def make_prompt(self, obj: PromptObject):
        system_prompt = "Compare the following two dataframes. "
        system_prompt += "They are similar if they are almost identical, or if they convey "
        system_prompt += "the same information about the nba_roster dataset. "
        system_prompt += "Respond with valid JSON {'explanation' : str, 'similar' : bool}"
        user_prompt = (
                f"========== Dataframe 1 =========\n{str(obj.data.get('df', 'None')).lower()}\n\n")
        user_prompt += (
                f"========== Dataframe 2 =========\n{str(obj.data['reference_df']).lower()}\n\n")
        user_prompt += f"Can you tell me if these dataframes are similar?"

        return {"system" : system_prompt, "user" : user_prompt}


## ------------------------------------------------------ ##
class EvaluationPipeline(GenerationPipeline):
    def __init__(self, args):
        super().__init__()
        self.query_stage = QueryStage(args.sql_model_name)
        self.score_stage = ScoreStage()


    def forward(self, x):
        x = self.query_stage(x)
        x = self.score_stage(x)

        return x

## ------------------------------------------------------ ##
async def run_eval(dataset, args):
    results = await run_evaluation_pipeline(dataset, args)
    print("Total results: ", len(results))

    return results

async def run_evaluation_pipeline(dataset, args):
    results = EvaluationPipeline(args).call(dataset)
    result_list = []

    pbar = tqdm(desc = "Saving results", unit = " results")

    async for result in results:
        result_list.append(result)
        pbar.update()

    return result_list

## ------------------------------------------------------ ##
def save_eval_results(results, args):
    base_path = "./data/results"
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    experiment_name = f"nba_sql_pipeline_{now}"
    experiment_dir = os.path.join(base_path, experiment_name)
    os.makedirs(os.path.join(base_path, experiment_name))

    args_file_name = f"{experiment_dir}/args.txt"

    with open(args_file_name, "w") as writer:
        pprint(args.__dict__, writer)


    def is_correct(r):
        if ((r.data["query_succeeded"] and r.data['is_matching']) or
            r.data["generated_query"] == r.data['sql']):
            return True

        return False


    results_file_name = f"{experiment_dir}/sql_results.jsonl"
    with jsonlines.open(results_file_name, "w") as writer:
        for result in results:
            if not is_correct(result):
                continue

            writer.write({"question" : result.data['question'],
                        "query" : result.data["generated_query"],
                        "query_succeeded" : result.data["query_succeeded"],
                        "reference_sql" : result.data['sql'],
                        "df" : str(result.data.get('df', 'None')),
                        "reference_df" : str(result.data['reference_df']),
                        'is_matching' : result.data['is_matching'],
                        'similar' : result.data['similar'], })

    results_file_name = f"{experiment_dir}/sql_errors.jsonl"

    with jsonlines.open(results_file_name, "w") as writer:
        for result in results:
            if is_correct(result):
                continue

            writer.write({"question" : result.data['question'],
                        "query" : result.data["generated_query"],
                        "query_succeeded" : result.data["query_succeeded"],
                        "df" : str(result.data.get('df', 'None')),
                        "reference_df" : str(result.data['reference_df']),
                        'is_matching' : result.data['is_matching'],
                        'similar' : result.data['similar'], })

    average_sql_succeeded = sum(
                            [result.data["query_succeeded"] for result in results]) / len(results)
    average_correct = sum(
            [result.data["query_succeeded"] and result.data['is_matching'] for result in results]
            ) / len(results)

    file_name = f"{experiment_dir}/summary.txt"

    with open(file_name, "w") as writer:
        print(f"Total size of eval dataset: {len(results)}", file = writer)
        print(f"Total size of eval dataset: {len(results)}")
        print(f"Percent Valid SQL Syntax: {average_sql_succeeded * 100}", file = writer)
        print(f"Percent Valid SQL Syntax: {average_sql_succeeded * 100}")
        print(f"Percent Correct SQL Query: {average_correct * 100}", file = writer)
        print(f"Percent Correct SQL Query: {average_correct * 100}")


## ------------------------------------------------------ ##
args = Args()
dataset = load_gold_dataset(args)
results = await run_eval(dataset, args)
save_eval_results(results, args)
