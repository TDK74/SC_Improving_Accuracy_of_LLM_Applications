import copy
import logging
import os
import random
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
from util.get_default_finetune_args import get_default_finetune_args
from util.get_schema import get_schema, get_schema_s
from util.load_dataset import get_dataset
from util.make_llama_3_prompt import make_llama_3_prompt
from util.setup_logging import setup_logging


_ = load_dotenv()

## ------------------------------------------------------ ##
logger = logging.getLogger(__name__)
engine = sqlite3.connect("./nba_roster.db")
setup_logging()

## ------------------------------------------------------ ##
class Args:
    def __init__(self,
                 max_examples = 100,
                 sql_model_name = "meta-llama/Meta-Llama-3-8B-Instruct",
                 gold_file_name = "gold-test-set.jsonl",
                 training_file_name = "generated_queries.jsonl",
                 num_to_generate = 10):
        self.sql_model_name = sql_model_name
        self.max_examples = max_examples
        self.gold_file_name = gold_file_name
        self.training_file_name = training_file_name
        self.num_to_generate = num_to_generate

## ------------------------------------------------------ ##
system = "You are an NBA analyst with 15 years of experience writing complex SQL queries.\n"
system += ("Consider a table called 'nba_roster' with the following schema (columns)\n")
system += get_schema_s()
system += "Consider the following questions, and queries used to answer them:\n"


## ------------------------------------------------------ ##
print(system)

## ------------------------------------------------------ ##
question = """What is the median weight in the NBA?"""
sql = "select CAST(SUBSTR(WT, 1, INSTR(WT, ' ')) as INTEGER) as percentile from nba_roster order\
        by percentile limit 1 offset (select count(*) from nba_roster) / 2;"
system += "Question: " + question + "\n"
system += "Query: " + sql + "\n"

## ------------------------------------------------------ ##
print(system)

## ------------------------------------------------------ ##
user = "Write two queries that are similar but different to those above.\n"
user += "Format the queries as a JSON object, i.e.\n"
user += '{ "explanation": str, "sql_query_1" : str, "sql_query_2": str }.\n'

## ------------------------------------------------------ ##
print(user)

## ------------------------------------------------------ ##
user += "First write an explanation of why you decided to write these new queries in about 3-5 \
        sentences, then write valid sqlite SQL queries for each of the 2 new queries. Make sure \
        each query is complete and ends with a ;\n"

## ------------------------------------------------------ ##
print(user)

## ------------------------------------------------------ ##
prompt = make_llama_3_prompt(user, system)

## ------------------------------------------------------ ##
llm = lamini.Lamini(model_name = "meta-llama/Meta-Llama-3-8B-Instruct")
result = llm.generate(prompt,
                    output_type = { "explanation" : "str", "sql_query_1" : "str",
                                    "sql_query_2" : "str" },
                    max_new_tokens = 200)
print(result)

## ------------------------------------------------------ ##
def check_sql_query(query):
    try:
        pd.read_sql(query, con = engine)

    except Exception as e:
        logger.debug(f"Error in SQL query: {e}")

        return False

    logger.info(f"SQL query {query} is valid")

    return True

## ------------------------------------------------------ ##
check_sql_query(result["sql_query_1"])

## ------------------------------------------------------ ##
check_sql_query(result["sql_query_2"])

## ------------------------------------------------------ ##
class ModelStage(GenerationNode):
    def __init__(self):
        super().__init__(model_name = "meta-llama/Meta-Llama-3-8B-Instruct", max_new_tokens = 300, )


    def generate(self,
                prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
                *args,
                **kwargs, ):
        prompt = self.add_template(prompt)

        results = super().generate(prompt,
                                    output_type = {"explanation" : "str", "sql_query_1" : "str",
                                                    "sql_query_2" : "str", },
                                    *args,
                                    **kwargs, )

        return results


    async def add_template(self, prompts):
        async for prompt in prompts:
            new_prompt = make_llama_3_prompt(**self.make_prompt(prompt.data))

            yield PromptObject(prompt = new_prompt, data = prompt.data)


    async def process_results(self, results):
        async for result in results:
            if result is None:
                continue

            if result.response is None:
                continue

            logger.info("=====================================")
            logger.info(f"Generated query 1: {result.response['sql_query_1']}")
            logger.info(f"Generated query 2: {result.response['sql_query_2']}")
            logger.info("=====================================")

            if self.check_sql_query(result.response["sql_query_1"]):
                new_result = PromptObject(prompt = "", data = copy.deepcopy(result.data))
                new_result.data.generated_sql_query = result.response["sql_query_1"]

                yield new_result

            if self.check_sql_query(result.response["sql_query_2"]):
                new_result = PromptObject(prompt = "", data = copy.deepcopy(result.data))
                new_result.data.generated_sql_query = result.response["sql_query_2"]

                yield new_result


    def make_prompt(self, data):
        system = "You are an NBA analyst with 15 years of experience writing complex SQL queries.\n"
        system += ("Consider a table called 'nba_roster' with the following schema (columns)\n")
        system += get_schema()
        system += "Consider the following questions, and queries used to answer them:\n"

        for example in data.sample:
            system += "Question: " + example["question"] + "\n"
            system += "Query: " + example["sql"] + "\n"

        user = "Write two queries that are similar but different to those above.\n"
        user += "Format the queries as a JSON object, i.e.\n"
        user += '{ "explanation": str, "sql_query_1" : str, "sql_query_2": str }.\n'

        user += "First write an explanation of why you decided to write these new queries in about \
                3-5 sentences, then write valid sqlite SQL queries for each of the 2 new queries.\
                Make sure each query is complete and ends with a ;\n"

        return {"system": system, "user": user}


    def check_sql_query(self, query):
        try:
            pd.read_sql(query, con = engine)

        except Exception as e:
            logger.debug(f"Error in SQL query: {e}")

            return False

        logger.info(f"SQL query {query} is valid")

        return True

## ------------------------------------------------------ ##
system = "You are an NBA analyst with 15 years of experience writing complex SQL queries.\n"
system += ("Consider a table called 'nba_roster' with the following schema (columns)\n")
system += get_schema() + "\n"
system += "Queries, and questions that they are used to answer:\n"

example_question = """What is the median weight in the NBA?"""
example_sql = "select CAST(SUBSTR(WT, 1, INSTR(WT,' ')) as INTEGER) as percentile from nba_roster\
                order by percentile limit 1 offset (select count(*) from nba_roster) / 2;"

system += "Question: " + example_question + "\n"
system += "Query: " + example_sql + "\n"

## ------------------------------------------------------ ##
generated_sql = result["sql_query_2"]

## ------------------------------------------------------ ##
user = "Now consider the following query.\n"
user += "Query: " + generated_sql + "\n"
user += "Write a question that this query could be used to answer.\n"

## ------------------------------------------------------ ##
user += "Format your response as a JSON object, i.e.\n"
user += '{ "explanation": str, "question": str }.\n'

user += "First write an explanation in about 3-5 sentences, then write a one sentence question.\n"

## ------------------------------------------------------ ##
prompt = make_llama_3_prompt(user, system)
result = llm.generate(prompt, output_type = { "explanation" : "str", "question" : "str" },
                    max_new_tokens = 200)
print(result)

## ------------------------------------------------------ ##
class QuestionStage(GenerationNode):
    def __init__(self):
        super().__init__(model_name = "meta-llama/Meta-Llama-3-8B-Instruct", max_new_tokens = 150, )

    def generate(self,
                prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
                *args,
                **kwargs, ):
        results = super().generate( prompt,
                                    output_type={"explanation" : "str", "question" : "str", },
                                    *args,
                                    **kwargs,
                                    )

        return results


    def preprocess(self, obj: PromptObject):
        new_prompt = make_llama_3_prompt(**self.make_question_prompt(obj.data))
        obj.prompt = new_prompt


    def make_question_prompt(self, data):
        system = "You are an NBA analyst with 15 years of experience writing complex SQL queries.\n"
        system += ("Consider a table called 'nba_roster' with the following schema (columns)\n")
        system += get_schema() + "\n"
        system += "Queries, and questions that they are used to answer:\n"

        for example in data.sample:
            system += "Query: " + example["sql"] + "\n"
            system += "Question: " + example["question"] + "\n"

        user = "Now consider the following query.\n"
        user += "Query: " + data.generated_sql_query + "\n"
        user += "Write a question that this query could be used to answer.\n"

        user += "Format your response as a JSON object, i.e.\n"
        user += '{"explanation" : str, "question" : str }.\n'

        user += "First write an explanation in about 3-5 sentences, then write a one sentence \
                question.\n"

        return {"system" : system, "user" : user}

## ------------------------------------------------------ ##
class QueryGenPipeline(GenerationPipeline):
    def __init__(self):
        super().__init__()
        self.model_stage = ModelStage()
        self.question_stage = QuestionStage()

    def forward(self, x):
        x = self.model_stage(x)
        x = self.question_stage(x)

        return x

## ------------------------------------------------------ ##
async def run_query_gen_pipeline(gold_queries):
    return QueryGenPipeline().call(gold_queries)

## ------------------------------------------------------ ##
all_examples = []

async def load_gold_queries(args):
    path = f"data/{args.gold_file_name}"

    with jsonlines.open(path) as reader:
        global all_examples

        all_examples = [obj for obj in reader]

    sample_count = args.num_to_generate
    sample_size = 3

    random.seed(42)

    for i in range(sample_count):
        example_sample = ExampleSample(random.sample(all_examples, sample_size), i)
        yield PromptObject(prompt = "", data = example_sample)


class ExampleSample:
    def __init__(self, sample, index):
        self.sample = sample
        self.index = index

## ------------------------------------------------------ ##
async def save_generation_results(results, args):
    path = f"data/training_data/{args.training_file_name}"

    pbar = tqdm(desc = "Saving results", unit = " results")

    with jsonlines.open(path, "w") as writer:

        async for result in results:
            writer.write({"question" : result.response["question"],
                        "sql" : result.data.generated_sql_query, })
            pbar.update()

        for example in all_examples:
            writer.write(example)
            pbar.update()

## ------------------------------------------------------ ##
args = Args()
gold_queries = load_gold_queries(args)
results = await run_query_gen_pipeline(gold_queries)
await save_generation_results(results, args)

## ------------------------------------------------------ ##
def make_question(obj):
    system = "You are an NBA analyst with 15 years of experience writing complex SQL queries.\n"
    system += "Consider the nba_roster table with the following schema:\n"
    system += get_schema() + "\n"
    system += ("Write a sqlite SQL query that would help you answer the following question:\n")
    user = obj["question"]

    return {"system" : system, "user" : user}

## ------------------------------------------------------ ##
args = Args()
llm = lamini.Lamini(model_name = "meta-llama/Meta-Llama-3-8B-Instruct")

## ------------------------------------------------------ ##
dataset = get_dataset(args, make_question)

## ------------------------------------------------------ ##
finetune_args = get_default_finetune_args()

## ------------------------------------------------------ ##
# llm.train(data_or_dataset_id = dataset, finetune_args = finetune_args, is_public = True, )

## ------------------------------------------------------ ##
llm = lamini.Lamini(model_name = "a5ebf1c4879569101f32444afae5adcafbfce9c5a6ed13035fd892147f7d59bc")

## ------------------------------------------------------ ##
question = """Who is the highest paid NBA player?"""
system = f"""You are an NBA analyst with 15 years of experience writing complex SQL queries.
        Consider the nba_roster table with the following schema:
        {get_schema()}

        Write a sqlite query to answer the following question. Follow instructions exactly"""
prompt = make_llama_3_prompt(question, system)
print("Question:\n", question)

## ------------------------------------------------------ ##
print("Answer:")
print(llm.generate(prompt, max_new_tokens = 200))

## ------------------------------------------------------ ##
query = "SELECT salary, name FROM nba_roster WHERE salary \
        != '--' ORDER BY CAST(REPLACE(REPLACE(salary, '$', ''), ',','') AS INTEGER) DESC LIMIT 1;"
df = pd.read_sql(query, con = engine)
print(df)

## ------------------------------------------------------ ##
class QueryStage(GenerationNode):
    def __init__(self, model_name):
        super().__init__(model_name = model_name, max_new_tokens = 300, )


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
            logger.info(f"Running SQL query '{obj.response['sqlite_query']}'")
            obj.data["generated_query"] = obj.response["sqlite_query"]
            df = pd.read_sql(obj.response["sqlite_query"], con = engine)
            obj.data['df'] = df
            logger.info(f"Got data: {df}")
            query_succeeded = True

        except Exception as e:
            logger.error(f"Failed to run SQL query: {obj.response['sqlite_query']}")

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
        system += ("Write a sqlite SQL query that would help you answer the following question.\
                    Make sure each query ends with a semicolon:\n")
        user = data["question"]

        return {"user" : user, "system" : system, }


class ScoreStage(GenerationNode):
    def __init__(self):
        super().__init__(model_name = "meta-llama/Meta-Llama-3-8B-Instruct", max_new_tokens = 150, )

    def generate(self,
                prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
                *args,
                **kwargs, ):
        results = super().generate(prompt,
                            output_type = {"explanation" : "str", "similar" : ["true", "false"]},
                            *args,
                            **kwargs, )

        return results


    def preprocess(self, obj: PromptObject):
        obj.prompt = make_llama_3_prompt(**self.make_prompt(obj))
        logger.info(f"Scoring Stage Prompt:\n{obj.prompt}")


    def postprocess(self, obj: PromptObject):
        obj.data['is_matching'] = self.is_matching(obj.data, obj.response)
        obj.data['explanation'] = obj.response["explanation"]
        obj.data['similar'] = obj.response["similar"] == "true"


    def is_matching(self, data, response):
        return (str(data.get('df', "None")).lower() == str(data['reference_df']).lower()
                or response['similar'] == "true")


    def make_prompt(self, obj: PromptObject):
        system_prompt = "Compare the following two dataframes. They are similar if they are almost \
                    identical, or if they convey the same information about the nba_roster dataset"
        system_prompt += "Respond with valid JSON {'explanation' : str, 'similar' : bool}"
        user_prompt = (f"========== Dataframe 1 =========\n{str(obj.data.get('df', 'None'))
                                                            .lower()}\n\n")
        user_prompt += (f"========== Dataframe 2 =========\n{str(obj.data['reference_df'])
                                                             .lower()}\n\n")
        user_prompt += f"Can you tell me if these dataframes are similar?"

        return {"system" : system_prompt, "user" : user_prompt}


async def run_eval(dataset, args):
    results = await run_evaluation_pipeline(dataset, args)

    print("Total results:", len(results))

    return results


async def run_evaluation_pipeline(dataset, args):
    results = EvaluationPipeline(args).call(dataset)

    result_list = []

    pbar = tqdm(desc = "Saving results", unit = " results")

    async for result in results:
        result_list.append(result)
        pbar.update()

    return result_list


class EvaluationPipeline(GenerationPipeline):
    def __init__(self, args):
        super().__init__()
        self.query_stage = QueryStage(args.sql_model_name)
        self.score_stage = ScoreStage()


    def forward(self, x):
        x = self.query_stage(x)
        x = self.score_stage(x)

        return x


def load_gold_dataset(args):
    path = f"data/{args.gold_file_name}"

    with jsonlines.open(path) as reader:

        for index, obj in enumerate(reversed(list(reader))):

            if index >= args.max_examples:
                break

            yield PromptObject(prompt = "", data = obj)


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
        if ((result.data["query_succeeded"] and result.data['is_matching']) or
            result.data["generated_query"] == result.data['sql']):
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
args = Args(sql_model_name = "a5ebf1c4879569101f32444afae5adcafbfce9c5a6ed13035fd892147f7d59bc")
dataset = load_gold_dataset(args)
results = await run_eval(dataset, args)
save_eval_results(results, args)

## ------------------------------------------------------ ##
question_set = set()
sql_set = set()


def is_not_valid_sql(question, sql):
    try:
        df = pd.read_sql(sql, con = engine)
        return False

    except Exception as e:
        return True


def has_null_in_sql_or_question(question, sql):
    return "null" in sql.lower() or "null" in question


def returns_empty_dataframe(question, sql):
    try:
        df = pd.read_sql(sql, con = engine)
        return "Empty" in str(df) or "None" in str(df)

    except Exception as e:
        return False


def uses_avg_on_ht_column(question, sql):
    return "avg(ht)" in sql.lower() or "avg(salary" in sql.lower()


filter_conditions = [is_not_valid_sql, has_null_in_sql_or_question,
                    returns_empty_dataframe, uses_avg_on_ht_column]


def training_semicolon(sql):
    if sql.strip()[-1] != ";":
        return sql.strip() + ";"

    return sql


with jsonlines.open("data/training_data/archive/generated_queries_large.jsonl", "r") as reader:

    with jsonlines.open("data/training_data/generated_queries_large_filtered.jsonl", "w") as writer:

        for r in reader:
            if r["question"] in question_set or r["sql"] in sql_set:
                continue

            question_set.add(r["question"])
            sql_set.add(r["sql"])

            if any(c(r['question'], r['sql']) for c in filter_conditions):
                continue

            sql = training_semicolon(r['sql'])
            writer.write({"question" : r["question"], "sql" : sql, })

## ------------------------------------------------------ ##
df = pd.read_sql("SELECT AVG(CAST(SUBSTR(WT, 1, INSTR(WT,' ')) as INTEGER)" \
                "FROM nba_roster WHERE WT!= 'NA') as median", con = engine)

## ------------------------------------------------------ ##
df = pd.read_sql("SELECT COLLEGE, COUNT(*) as count FROM nba_roster WHERE COLLEGE!= '--' \
                GROUP BY COLLEGE ORDER BY count DESC LIMIT 1", con = engine)
print(df)

## ------------------------------------------------------ ##
llm = lamini.Lamini(model_name = "63fd73a775daf24216b46c680a1e963a8d1e02b21bca43fcea6c26737d2e887e")

## ------------------------------------------------------ ##
question = """What is the median age of the Chicago Bulls?"""
system = f"""You are an NBA analyst with 15 years of experience writing complex SQL queries. \
            Consider the nba_roster table with the following schema:
            {get_schema()}

            Write a sqlite query to answer the following question. Follow instructions exactly"""
prompt = make_llama_3_prompt(question, system)
print("Question:\n", question)

print("Answer:")
sql = llm.generate(prompt, max_new_tokens = 200)
print(sql)

## ------------------------------------------------------ ##
df = pd.read_sql(sql, con = engine)
print(df)

## ------------------------------------------------------ ##
args = Args(training_file_name = "archive/generated_queries_v2_large_filtered_cleaned.jsonl")

## ------------------------------------------------------ ##
llm = lamini.Lamini(model_name = "meta-llama/Meta-Llama-3-8B-Instruct")

## ------------------------------------------------------ ##
dataset = get_dataset(args, make_question)
finetune_args = get_default_finetune_args()

## ------------------------------------------------------ ##
# llm.train(data_or_dataset_id = dataset, finetune_args = finetune_args, is_public = True, )

## ------------------------------------------------------ ##
class QueryStage(GenerationNode):
    def __init__(self, model_name):
        super().__init__(model_name = model_name, max_new_tokens = 300, )


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
            logger.info(f"Running SQL query '{obj.response['sqlite_query']}'")
            obj.data["generated_query"] = obj.response["sqlite_query"]
            df = pd.read_sql(obj.response["sqlite_query"], con = engine)
            obj.data['df'] = df
            logger.info(f"Got data: {df}")
            query_succeeded = True

        except Exception as e:
            logger.error(f"Failed to run SQL query: {obj.response['sqlite_query']}")

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


class ScoreStage(GenerationNode):
    def __init__(self):
        super().__init__(model_name = "meta-llama/Meta-Llama-3-8B-Instruct", max_new_tokens = 150, )


    def generate(self,
                prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
                *args,
                **kwargs, ):
        results = super().generate(prompt,
                            output_type = {"explanation" : "str", "similar" : ["true", "false"]},
                            *args,
                            **kwargs, )

        return results


    def preprocess(self, obj: PromptObject):
        obj.prompt = make_llama_3_prompt(**self.make_prompt(obj))
        logger.info(f"Scoring Stage Prompt:\n{obj.prompt}")


    def postprocess(self, obj: PromptObject):
        obj.data['is_matching'] = self.is_matching(obj.data, obj.response)
        obj.data['explanation'] = obj.response["explanation"]
        obj.data['similar'] = obj.response["similar"] == "true"


    def is_matching(self, data, response):
        return (str(data.get('df',"None")).lower() == str(data['reference_df']).lower()
                or response['similar'] == "true")

    def make_prompt(self, obj: PromptObject):
        system_prompt = "Compare the following two dataframes. They are similar if they are almost \
                    identical, or if they convey the same information about the nba_roster dataset"
        system_prompt += "Respond with valid JSON {'explanation' : str, 'similar' : bool}"
        user_prompt = (f"========== Dataframe 1 =========\n{str(obj.data.get('df','None'))
                                                            .lower()}\n\n")
        user_prompt += (f"========== Dataframe 2 =========\n{str(obj.data['reference_df'])
                                                            .lower()}\n\n")
        user_prompt += f"Can you tell me if these dataframes are similar?"

        return {"system" : system_prompt, "user" : user_prompt}


async def run_eval(dataset, args):
    results = await run_evaluation_pipeline(dataset, args)
    print("Total results:", len(results))

    return results


async def run_evaluation_pipeline(dataset, args):
    results = EvaluationPipeline(args).call(dataset)

    result_list = []

    pbar = tqdm(desc = "Saving results", unit = " results")

    async for result in results:
        result_list.append(result)
        pbar.update()

        return result_list


class EvaluationPipeline(GenerationPipeline):
    def __init__(self, args):
        super().__init__()
        self.query_stage = QueryStage(args.sql_model_name)
        self.score_stage = ScoreStage()


    def forward(self, x):
        x = self.query_stage(x)
        x = self.score_stage(x)

        return x


def load_gold_dataset(args):
    path = f"data/{args.gold_file_name}"

    with jsonlines.open(path) as reader:

        for index, obj in enumerate(reversed(list(reader))):

            if index >= args.max_examples:
                break

            yield PromptObject(prompt = "", data = obj)


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
        if ((result.data["query_succeeded"] and result.data['is_matching']) or
            result.data["generated_query"] == result.data['sql']):
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

    average_sql_succeeded = sum([result.data["query_succeeded"] for result in results]
                                ) / len(results)
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
args = Args(sql_model_name = "3f7e740c0ea2227631a30d293b51564ad1b80727c3768a3b136fbae93170c1e2",
            gold_file_name = 'gold-test-set-v2.jsonl')
dataset = load_gold_dataset(args)
results = await run_eval(dataset, args)
save_eval_results(results, args)
