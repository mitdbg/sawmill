import pandas as pd
import numpy as np
import networkx as nx
from pgmpy.estimators import PC, HillClimbSearch, ExhaustiveSearch
from pgmpy.estimators import K2Score
from pgmpy.base import DAG
from openai import OpenAI
from tqdm.auto import tqdm
from datetime import datetime
from typing import Optional, Tuple
from .tag_utils import TagUtils


class CausalDiscoverer:
    """
    Provides various methods for automatic causal discovery based on a dataframe.

    Within Sawmill, the expectation is that the passed dataframe will contain the prepared variables.
    """

    @staticmethod
    def _pgmpy_dag_to_digraph(dag: DAG) -> nx.DiGraph:
        """
        Converts a pgmpy DAG to a networkx DiGraph.

        Parameters:
            dag: The pgmpy DAG.

        Returns:
            The networkx DiGraph.
        """

        return nx.DiGraph(dag.edges())

    @staticmethod
    def pc(df: pd.DataFrame, max_cond_vars: int = 3) -> nx.DiGraph:
        """
        Runs the PC algorithm on a dataframe.

        Parameters:
            df: The dataframe on which to run the PC algorithm.
            max_cond_vars: The maximum number of conditioning variables to use.

        Returns:
            The causal graph learned by the PC algorithm.
        """

        pc = PC(data=df)
        model = pc.estimate(variant="parallel", max_cond_vars=max_cond_vars)
        return CausalDiscoverer._pgmpy_dag_to_digraph(model)

    @staticmethod
    def hill_climb(df: pd.DataFrame) -> nx.DiGraph:
        """
        Runs the hill climb algorithm on a dataframe.

        Parameters:
            df: The dataframe on which to run the hill climb algorithm.

        Returns:
            The causal graph learned by the hill climb algorithm.
        """

        scoring_method = K2Score(data=df)
        hcs = HillClimbSearch(data=df)
        model = hcs.estimate(
            scoring_method=scoring_method, max_indegree=4, max_iter=int(1e4)
        )
        return CausalDiscoverer._pgmpy_dag_to_digraph(model)

    @staticmethod
    def exhaustive(df: pd.DataFrame) -> nx.DiGraph:
        """
        Runs the exhaustive search algorithm on a dataframe.

        Parameters:
            df: The dataframe on which to run the exhaustive search algorithm.

        Returns:
            The causal graph learned by the exhaustive search algorithm.
        """

        scoring_method = K2Score(data=df)
        exh = ExhaustiveSearch(data=df, complete_samples_only=False)
        model = exh.estimate()
        return CausalDiscoverer._pgmpy_dag_to_digraph(model)

    @staticmethod
    def gpt(
        data_df: pd.DataFrame,
        model: str = "gpt-3.5-turbo",
        vars_df: Optional[pd.DataFrame] = None,
    ) -> nx.DiGraph:
        """
        Consults GPT to determine the causal graph of the variables in the dataframe.

        Parameters:
            data_df: The dataframe based on which to construct a causal graph.
            model: The GPT model to use.
            vars_df: The dataframe containing the variable names and tags.

        Returns:
            The causal graph learned by consulting GPT.
        """

        # Open a file for logging, with the model and the timestamp in the name
        log_file = open(
            f"/../../evaluation/gpt-logs/{model}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt",
            "w",
        )

        client = OpenAI()
        graph = nx.DiGraph()

        for i in tqdm(
            range(len(data_df.columns)), desc="Outer edge-finding loop using GPT..."
        ):
            for j in range(i + 1, len(data_df.columns)):
                var_a = data_df.columns[i]
                var_b = data_df.columns[j]

                example_rows = data_df[[var_a, var_b]].dropna().sample(3)
                examples_a = ", ".join(str(x) for x in example_rows[var_a].tolist())
                examples_b = ", ".join(str(x) for x in example_rows[var_b].tolist())

                tag_a = (
                    var_a
                    if vars_df is None
                    else TagUtils.get_tag(vars_df, var_a, "prepared")
                )
                tag_b = (
                    var_b
                    if vars_df is None
                    else TagUtils.get_tag(vars_df, var_b, "prepared")
                )

                # Define the messages to send to the model
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for causal reasoning.",
                    },
                    {
                        "role": "user",
                        "content": f"""Which cause-and-effect relationship is more likely? """
                        f"""A. changing {tag_a} causes a change in {tag_b}. """
                        f"""B. changing {tag_b} causes a change in {tag_a}. """
                        f"""C. Neither of the two. """
                        f""" Here are some example values of {tag_a} : [{examples_a}]"""
                        f""" Here are the corresponding values of {tag_b} : [{examples_b}]"""
                        """Let's work this out in a step by step way to be sure that we have the right answer. """
                        """Then provide your ﬁnal answer within the tags <Answer>A/B/C</Answer>.""",
                    },
                ]

                reply = (
                    client.chat.completions.create(model=model, messages=messages)
                    .choices[0]
                    .message.content
                )

                # Log the messages and the reply
                log_file.write(f"{datetime.now()}\n")
                log_file.write("Messages:\n")
                for message in messages:
                    log_file.write(f"{message['role']}: {message['content']}\n")
                log_file.write("----------------\n")
                log_file.write(f"Reply: {reply}\n\n")
                log_file.write("================\n")
                log_file.flush()

                # Find the part of the reply that contains the answer
                start_idx = reply.find("<Answer>") + len("<Answer>")
                end_idx = reply.find("</Answer>")
                answer = reply[start_idx:end_idx]

                # Add the edge to the graph
                if answer == "A":
                    graph.add_edge(var_a, var_b)
                elif answer == "B":
                    graph.add_edge(var_b, var_a)
        log_file.close()
        return graph

    @staticmethod
    def gpt_baseline(
        data_df: pd.DataFrame,
        file_tag: str,
        outcome: str = None,
        vars_df: Optional[pd.DataFrame] = None,
        k: int = 10,
        model: str = "gpt-4-1106-preview",
    ) -> Tuple[str, float]:
        """
        Uses GPT to find the most likely causes of a variable in a dataframe, as a replacement for candidate cause exploration.

        Parameters:
            data_df: The dataframe based on which to find the most likely causes.
            file_tag: The tag to use in the log file name.
            outcome: The variable for which to find the most likely causes.
            vars_df: The dataframe containing the variable names and tags.
            k: The number of example values to show for each variable.
            model: The GPT model to use.

        Returns:
            The name of the log file and the time elapsed.
        """

        start_time = datetime.now()

        # Open a file for logging, with the model and the timestamp in the name
        filename = f"../../evaluation/gpt-logs/{file_tag}-{model}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt"
        log_file = open(
            filename,
            "w",
        )

        client = OpenAI()
        tag_outcome = (
            outcome
            if vars_df is None
            else TagUtils.tag_of(vars_df, outcome, "prepared")
        )

        # Define the messages to send to the model
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant for causal reasoning.",
            },
            {
                "role": "user",
                "content": f"""Below is a list of variable names and some example values for each. """
                f"""The lists are sorted in compatible ways, so that elements in the same position correspond to the same entity. """
                f"""I want you to find the (up to 25) most likely causes for variable '{tag_outcome}' and return them as a ranked list. """
                """I understand that you may think this is speculative, but I want you to do your best to come up with such a list ALWAYS. """
                """I will interpret any results you give me knowing that you may not be sure about them. """
                """I also want you to create a causal directed acyclic graph out of the variables and return all the edges, one per line. """
                f"""Make sure the causal DAG includes the variable '{tag_outcome}' and is consistent with the ranked list of causes. """
                """Again, I understand that you may think this is speculative, but I want you to do your best to come up with such a graph ALWAYS. """
                """I will interpret any results you give me knowing that you may not be sure about them. """
                """Here are the variables: """
                f"""{', '.join([f'{TagUtils.tag_of(vars_df, v, "prepared")}: [{", ".join(str(x) for x in data_df[v].tolist()[:k])}]' for v in data_df.columns])}""",
            },
        ]

        reply = (
            client.chat.completions.create(model=model, messages=messages)
            .choices[0]
            .message.content
        )
        end_time = datetime.now()
        elapsed = round((end_time - start_time).total_seconds(), 6)

        # Log the messages and the reply
        log_file.write(f"{datetime.now()}\n")
        log_file.write("Messages:\n")
        for message in messages:
            log_file.write(f"{message['role']}: {message['content']}\n")
        log_file.write("----------------\n")
        log_file.write(f"Reply: {reply}\n\n")
        log_file.write(f"Time elapsed: {elapsed}\n")
        log_file.write("================\n")
        log_file.flush()
        log_file.close()

        return filename, elapsed
