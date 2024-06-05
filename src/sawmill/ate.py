import contextlib
import pandas as pd
import networkx as nx
from typing import Optional, Any
from .tag_utils import TagUtils
from dowhy import CausalModel
from .types import Types
import os
import pickle
from .clustering_params import ClusteringParams
from .regression import Regression
from itertools import combinations
from .variable_name.prepared_variable_name import PreparedVariableName
from .printer import Printer
from tqdm.auto import tqdm
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from .edge_occurrence_tree import EdgeOccurrenceTree
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from .variable_name.prepared_variable_name import PreparedVariableName
import pandas as pd
from src.sawmill.printer import Printer
import numpy as np
from tqdm.auto import tqdm
import networkx as nx
import pickle
import os
from .pickler import Pickler
from eccs.eccs import ECCS


class Pruner:
    LASSO_DEFAULT_ALPHA = 0.1
    LASSO_DEFAULT_MAX_ITER = 100000

    """
    A collection of pruning functions for prepared variables,
    used for pruning and candidate suggestion.
    """

    @staticmethod
    def prune_with_lasso(
        data: pd.DataFrame,
        outcome_cols: list[str],
        alpha: float = LASSO_DEFAULT_ALPHA,
        max_iter: int = LASSO_DEFAULT_MAX_ITER,
        top_n: int = 0,
        ignore: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Prune variables using Lasso regression.

        Parameters:
            data: The dataframe containing the data.
            outcome_cols: The names of the target variables.
            alpha: The Lasso regularization parameter.
            max_iter: The maximum number of iterations for Lasso.
            top_n: The number of variables to return. If 0, return all variables.

        Returns:
            The names of the variables that Lasso identified as impactful, optionally
            limited to the top `n` variables by absolute coefficient.
        """

        # TODO: do this properly wherever this is called
        outcome_col = outcome_cols[0]

        # Separate the target variable and predictor variables.
        # Optionally, do not consider variables already in the graph.
        y = data[outcome_cols]
        drop_cols = [] if ignore is None else ignore
        to_ignore = outcome_cols
        drop_cols.extend(to_ignore)

        # Do not consider variables with the same base variable as an ignored variable.
        for v in to_ignore:
            vp = PreparedVariableName(v)
            if vp.base_var() != "TemplateId":
                drop_cols.extend([c for c in data.columns if vp.base_var() in c])
        drop_cols = list(set(drop_cols))

        # Iterate until multiple prepared variables with the same base variable are eliminated.
        done = False

        while not done:
            Printer.printv(f"Variables that Lasso will ignore: {drop_cols}")
            X = data.drop(drop_cols, axis=1)
            X_cols = X.columns
            if X.empty:
                return []

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            # Fit a Lasso model to the data
            lasso = Lasso(alpha=alpha, max_iter=max_iter)
            lasso.fit(X, y)
            Printer.printv(f"Lasso coefficients : {lasso.coef_}")
            Printer.printv(f"Scale: {scaler.scale_}")
            final_coefs = lasso.coef_ / scaler.scale_
            abs_coefs = np.abs(final_coefs)
            Printer.printv(f"Lasso coefficients unscaled: {final_coefs}")

            # Mask for nonzero elements
            nonzero_mask = final_coefs != 0

            # Mask for top n largest elements by absolute value
            # Create an array of False values with the same shape as the coefficients
            top_n_mask = [False] * len(final_coefs)
            for i in np.argsort(abs_coefs)[-top_n:]:
                top_n_mask[i] = True

            # Retrieve columns based on conditions above
            selected_names = list(X_cols[nonzero_mask & top_n_mask])

            # Only keep one aggregate per variable
            d = set()
            done = True
            for var in selected_names:
                base_var = PreparedVariableName(var).base_var()
                if base_var in d:
                    drop_cols.append(var)
                    done = False
                else:
                    d.add(base_var)

        Printer.printv("Lasso identified the following impactful variables:")
        Printer.printv(selected_names)

        return selected_names

    @staticmethod
    def prune_with_triangle(
        data: pd.DataFrame,
        vars: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        work_dir: str,
        top_n: int = 0,
        force: bool = False,
    ) -> list[str]:
        """
        Prune variables using triangle method.

        Parameters:
            data: The dataframe containing the data.
            vars: The dataframe containing the variables.
            treatment_col: The name of the treatment variable.
            outcome_col: The name of the outcome variable.
            work_dir: The directory to store intermediate files in.
            top_n: The number of variables to return. If 0, return all variables.
            force: Whether to force recalculation of the triangle method.

        Returns:
            The names of the variables that triangle method identified as impactful, optionally
            limited to the top `n` variables.
        """

        # Check whether we can use pre-calculated results
        filename = os.path.join(
            work_dir, f"pickles/triangle_dags/{treatment_col}_{outcome_col}.pkl"
        )
        if os.path.isfile(filename) and not force:
            df = pickle.load(open(filename, "rb"))
            print("Found pickled file")
            return list(df.index[:top_n].values)

        Printer.printv("Starting to prune using triangle method")
        max_diffs = {}
        base_ate = ATECalculator.get_ate_and_confidence(
            data, vars, treatment_col, outcome_col, calculate_std_error=False
        )["ATE"]

        for var in tqdm(data.columns, "Processing triangle dags"):
            if var == treatment_col or var == outcome_col:
                continue

            # Construct the graphs to consider
            graphs = []
            # Second cause
            graphs.append(
                nx.DiGraph([(treatment_col, outcome_col), (var, outcome_col)])
            )
            # Confounder
            graphs.append(
                nx.DiGraph(
                    [
                        (treatment_col, outcome_col),
                        (var, treatment_col),
                        (var, outcome_col),
                    ]
                )
            )
            # Mediator with direct path
            graphs.append(
                nx.DiGraph(
                    [
                        (treatment_col, outcome_col),
                        (treatment_col, var),
                        (var, outcome_col),
                    ]
                )
            )
            # Mediator without direct path
            graphs.append(nx.DiGraph([(treatment_col, var), (var, outcome_col)]))

            # Calculate the corrsponding ATEs
            ates = [base_ate]
            for G in graphs:
                try:
                    ates.append(
                        ATECalculator.get_ate_and_confidence(
                            data,
                            vars,
                            treatment_col,
                            outcome_col,
                            graph=G,
                            calculate_std_error=False,
                        )["ATE"]
                    )
                except:
                    pass
            max_diffs[var] = max(ates) - min(ates)
        max_diffs = max_diffs
        df = pd.DataFrame.from_dict(max_diffs, orient="index", columns=["max_diff"])
        df = df.sort_values(by="max_diff", ascending=False)

        Pickler.dump(df, filename)

        return list(df.index[:top_n].values)


class ATECalculator:
    """
    A class to calculate ATEs and determine the impact of adding/removing/reversing DAG edges
    on these calculations.
    """

    @staticmethod
    def get_ate_and_confidence(
        data: pd.DataFrame,
        vars: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounder: Optional[str] = None,
        graph: Optional[nx.DiGraph] = None,
        calculate_p_value: bool = True,
        calculate_std_error: bool = True,
        get_estimand: bool = False,
    ) -> dict[str, Any]:
        """
        Calculate the ATE of `treatment` on `outcome`, alongside confidence measures.

        Parameters:
            treatment: The name or tag of the treatment variable.
            outcome: The name or tag of the outcome variable.
            confounder: The name or tag of a confounder variable. If specified, overrides the current partial
                causal graph in favor of a three-node graph with `treatment`, `outcome` and `confounder`.
            graph: The graph to be used for causal analysis. If not specified, a two-node graph with just
                `treatment` and `outcome` is used.
            calculate_p_value: Whether to calculate the P-value of the ATE.
            calculate_std_error: Whether to calculate the standard error of the ATE.
            get_estimand: Whether to return the estimand used to calculate the ATE, as part of the returned dictionary.

        Returns:
            A dictionary containing the ATE of `treatment` on `outcome`, alongside confidence measures. If
            `get_estimand` is True, the estimand used to calculate the ATE is also returned.
        """

        # If the user provided the tag of any variable, retrieve their names
        treatment = TagUtils.name_of(vars, treatment, "prepared")
        outcome = TagUtils.name_of(vars, outcome, "prepared")
        if confounder is not None:
            confounder = TagUtils.name_of(vars, confounder, "prepared")

        # Should the effects be calculated based on the current partial causal graph,
        # some other graph provided as a function parameter,
        # or on an ad-hoc subset relevant for the question at hand?
        if graph is None:
            graph = nx.DiGraph()
            graph.add_node(treatment)
            graph.add_node(outcome)
            graph.add_edge(treatment, outcome)

            if confounder is not None:
                graph.add_node(confounder)
                graph.add_edge(confounder, outcome)
                graph.add_edge(confounder, treatment)

        # Use dowhy to get the ATE, P-value and standard error.
        with open("/dev/null", "w+") as f:
            try:
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    model = CausalModel(
                        data=data[list(graph.nodes)],
                        treatment=treatment,
                        outcome=outcome,
                        graph=nx.nx_pydot.to_pydot(graph).to_string(),
                    )
                    identified_estimand = model.identify_effect(
                        proceed_when_unidentifiable=True
                    )
                    estimate = model.estimate_effect(
                        identified_estimand,
                        method_name="backdoor.linear_regression",
                        test_significance=True,
                    )
                    p_value = (
                        estimate.test_stat_significance()["p_value"].astype(float)[0]
                        if calculate_p_value
                        else None
                    )
                    stderr = (
                        estimate.get_standard_error() if calculate_std_error else None
                    )
                    d = {
                        "ATE": float(estimate.value),
                        "P-value": p_value,
                        "Standard Error": stderr,
                    }
                    if get_estimand:
                        d["Estimand"] = identified_estimand
                    return d
            except:
                raise ValueError

    @staticmethod
    def challenge_ate(
        data: pd.DataFrame,
        vars: pd.DataFrame,
        true_graph: Optional[nx.DiGraph],
        treatment: str,
        outcome: str,
        work_dir: str,
        num_outputs: int = 10,
        method: str = "step",
        cp: Optional[ClusteringParams] = None,
    ) -> pd.DataFrame:
        """
        Identify a ranked list of up to `num_outputs` possible edges among variables in the prepared log
        which, if set to a different state (i.e. included, reversed or omitted) would most noticeably
        impact the ATE of `treatment` on `outcome`.

        Parameters:
            data: The dataframe containing the data.
            vars: The dataframe containing information about the variables.
            true_graph: The starting graph to be used for causal analysis.
            treatment: The name or tag of the treatment variable.
            outcome: The name or tag of the outcome variable.
            work_dir: The directory to store intermediate files in.
            num_outputs: The maximum number of candidate changes to output.
            method: The method to use for ATE calculation. Can be one of: "step", "clustering", 
                "eccs-singleedit", "eccs-heuristicedit" or "eccs-adjsetedit".
            cp: The parameters to use for clustering. Only used if `method` is "clustering".
        Returns:
            A dataframe containing the edge changes that would most impact the ATE.
        """

        if method == "step":
            challenger = StepATEChallenger(
                data, vars, true_graph, treatment, outcome, num_outputs
            )
            return challenger.challenge()
        elif method == "clustering":
            cp = cp if cp is not None else ClusteringParams()
            challenger = ClusteringATEChallenger(
                data, vars, true_graph, treatment, outcome, work_dir, num_outputs, cp
            )
            return challenger.challenge()
        elif method.startswith("eccs"):
            challenger = ECCSATEChallenger(
                data, vars, true_graph, treatment, outcome, num_outputs, method
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def _is_acceptable(graph: nx.DiGraph, treatment: str, outcome: str) -> bool:
        """
        Check if a graph is acceptable for ATE calculation. A graph is acceptable
        if it is a DAG, contains the treatment and outcome variables, and has a directed path
        from the treatment to the outcome.

        Parameters:
            graph: The graph to be checked.
            treatment: The name or tag of the treatment variable.
            outcome: The name or tag of the outcome variable.

        Returns:
            Whether the graph is acceptable for ATE calculation.
        """
        return (
            nx.is_directed_acyclic_graph(graph)
            and graph.has_node(treatment)
            and graph.has_node(outcome)
            and nx.has_path(graph, treatment, outcome)
        )


class StepATEChallenger:
    """
    A class to calculate edge changes impactful to an ATE calculation using the step method.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        vars: pd.DataFrame,
        true_graph: Optional[nx.DiGraph],
        treatment: str,
        outcome: str,
        num_outputs: int = 10,
    ) -> None:
        """
        Initializes a StepATEChallenger.

        Parameters:
            data: The dataframe containing the data.
            vars: The dataframe containing information about the variables.
            true_graph: The starting graph to be used for causal analysis.
            treatment: The name or tag of the treatment variable.
            outcome: The name or tag of the outcome variable.
            num_outputs: The number of candidate changes to output.
        """

        self.data = data
        self.vars = vars
        self.true_graph = true_graph
        self.treatment = TagUtils.name_of(self.vars, treatment, "prepared")
        self.outcome = TagUtils.name_of(self.vars, outcome, "prepared")
        self.num_outputs = num_outputs

    def _enumerate_graphs(self) -> tuple[list[dict[str, str]], list[nx.DiGraph]]:
        """
        Enumerate graphs that are one edge different from the current graph,
        either by edge inclusion or by edge direction.
        """

        changes = []
        graphs = []
        for edge in self.true_graph.edges:
            # Remove the edge
            graph = self.true_graph.copy()
            graph.remove_edge(*edge)
            graphs.append(graph)
            changes.append({"Source": edge[0], "Target": edge[1], "Change": "Remove"})

            # Reverse the edge
            graph = self.true_graph.copy()
            graph.remove_edge(*edge)
            graph.add_edge(*reversed(edge))
            graphs.append(graph)
            changes.append({"Source": edge[0], "Target": edge[1], "Change": "Reverse"})

        # Add all possible edges
        for node in self.true_graph.nodes:
            for other_node in self.vars["Name"].values.tolist():
                if node != other_node and not self.true_graph.has_edge(
                    node, other_node
                ):
                    graph = self.true_graph.copy()
                    graph.add_edge(node, other_node)
                    graphs.append(graph)
                    changes.append(
                        {"Source": node, "Target": other_node, "Change": "Add"}
                    )

        print(f"Enumerated {len(graphs)} graphs with {len(graph.nodes)} nodes.")
        return changes, graphs

    def challenge(self) -> pd.DataFrame:
        """
        Calculate the ATE of `treatment` on `outcome` for all possible 1-step
        edge changes to `true_graph`, keeping the set of nodes the same.

        Returns:
            A dataframe containing the edge changes that would most impact the ATE.
        """

        changes, graphs = self._enumerate_graphs()
        baseline_ate = ATECalculator.get_ate_and_confidence(
            self.data, self.vars, self.treatment, self.outcome, graph=self.true_graph
        )
        graph_stats = []
        for i, graph in tqdm(enumerate(graphs)):
            if not ATECalculator._is_acceptable(graph, self.treatment, self.outcome):
                continue
            d = ATECalculator.get_ate_and_confidence(
                self.data, self.vars, self.treatment, self.outcome, graph=graph
            )
            d["Source"] = changes[i]["Source"]
            d["Source Tag"] = TagUtils.get_tag(self.vars, d["Source"], "prepared")
            d["Target"] = changes[i]["Target"]
            d["Target Tag"] = TagUtils.get_tag(self.vars, d["Target"], "prepared")
            d["Change"] = changes[i]["Change"]

            graph_stats.append(d.copy())

        graphs_df = pd.DataFrame(graph_stats)
        graphs_df["Baseline ATE"] = baseline_ate["ATE"]
        graphs_df["ATE Ratio"] = graphs_df.apply(
            lambda row: max(
                abs(row["ATE"] / row["Baseline ATE"]),
                abs(row["Baseline ATE"] / row["ATE"]),
            ),
            axis=1,
        )
        graphs_df.sort_values(by="ATE Ratio", ascending=False, inplace=True)

        column_order = [
            "Source",
            "Source Tag",
            "Target",
            "Target Tag",
            "Change",
            "ATE",
            "Baseline ATE",
            "ATE Ratio",
        ]

        return graphs_df[column_order].head(self.num_outputs)


class DendrogramRenderer:
    """
    A class to hold information for rendering a dendrogram.
    """

    def __init__(
        self, linkage: np.ndarray, num_clusters: int, llf: Types.LeafLabelingFunction
    ) -> None:
        """
        Initializes a DendrogramRenderer object.

        Parameters:
            linkage: linkage matrix as returned by scipy.cluster.hierarchy.linkage
            num_clusters: max num clusters to render
            llf: leaf labeling function, from leaf index to leaf label.
        """
        self.linkage = linkage
        self.num_cluster = num_clusters
        self.llf = llf


class ClusteringATEChallenger:
    """
    A class to calculate edge changes impactful to an ATE calculation using the clustering method.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        vars: pd.DataFrame,
        true_graph: Optional[nx.DiGraph],
        treatment: str,
        outcome: str,
        work_dir: str,
        num_outputs: int = 10,
        cp: Optional[ClusteringParams] = None,
    ) -> None:
        """
        Initializes a ClusteringATEChallenger.

        Parameters:
            data: The dataframe containing the data.
            vars: The dataframe containing information about the variables.
            true_graph: The starting graph to be used for causal analysis.
            treatment: The name or tag of the treatment variable.
            outcome: The name or tag of the outcome variable.
            work_dir: The directory to store intermediate files in.
            num_outputs: The number of candidate changes to output.
            cp: The parameters to use for clustering.
        """

        self.data = data
        self.vars = vars
        self.true_graph = true_graph
        self.treatment = TagUtils.name_of(vars, treatment, "prepared")
        self.outcome = TagUtils.name_of(vars, outcome, "prepared")
        self.work_dir = work_dir
        self.num_outputs = num_outputs
        self.cp = cp
        if self.cp is None:
            self.cp = ClusteringParams()

    def challenge(self) -> pd.DataFrame:
        """
        Use clustering to identify classes of ATEs based on presence/absence of specific
        important edges, and return the most impactful edge changes to transition between
        these classes.

        Returns:
            A dataframe containing the edge changes that would most impact the ATE.
        """

        filename = os.path.join(
            self.work_dir,
            f"pickles/effects/{self.treatment}_{self.outcome}_{self.cp.var_pruning_method}_{self.cp.triangle_n}.pkl",
        )

        # Check whether we can use pre-calculated results
        if os.path.isfile(filename) and not self.cp.force:
            self.effects = Pickler.load(filename)
        else:
            nodes_to_consider = self._find_nodes_to_consider()
            edges_to_consider = self._find_edges_to_consider(nodes_to_consider)
            dags_to_consider = self._find_dags_to_consider(edges_to_consider)
            self.effects = self._calculate_ates(dags_to_consider)
            Pickler.dump(self.effects, filename)

        # Cluster graphs by ATE
        self.dendrogram = self._cluster()
        Printer.printv(
            f"Successfully clustered ATEs into {self.num_clusters} with means {[val[0] for val in self.cluster_data.values()]}\n"
        )

        # Determine outliers
        self.tree.count_edge_occurrences(self.treatment, self.outcome, self.true_graph)
        self.tree.calculate_edge_expectancy()
        self.tree.find_outliers_in_tree(self.cp.threshold)
        self.edge_counts, self.outliers = self.tree.find_outliers_per_cluster(
            self.true_graph
        )

        for cluster, outlier_dict in self.outliers.items():
            Printer.printv(
                f"""For cluster centered around {self.cluster_data[cluster][0]:.3f} with {self.cluster_data[cluster][1]} points, """
                """the following edges were outliers with the corresponding frequency over/under expectation:"""
            )
            for outlier, percentage in outlier_dict.items():
                Printer.printv(f"{outlier}: {percentage * 100:.2f}%")

        # self.scored_edges = self._score_edges()
        self._display_important_edges()

    def _find_nodes_to_consider(self) -> list[str]:
        """
        Find the nodes to consider when enumerating DAGs under the clustering method.

        Returns:
            A list of nodes to consider.
        """

        if self.cp.var_pruning_method is None:
            return self.vars["Name"].values.tolist()

        # Remove variables related to treatment and outcome
        nodes_to_consider = [
            var
            for var in vars["Name"].values.tolist()
            if self.treatment not in var and self.outcome not in var
        ]

        # Remove variables that are already in the DAG or that are timestamp variables
        if self.true_graph is not None:
            nodes_to_consider = [
                var for var in nodes_to_consider if var not in self.true_graph.nodes
            ]
        if self.cp.ignore_ts:
            nodes_to_consider = [
                var for var in nodes_to_consider if "Timestamp" not in var
            ]

        # Prune aggregates
        nodes_to_consider = self._prune_aggregates(nodes_to_consider)

        # Prune further using the given method
        if self.cp.var_pruning_method_type == "lasso":
            nodes_to_consider = Regression.prune_with_lasso(
                self.data, [self.outcome], top_n=self.cp.n
            )
        elif self.cp.var_pruning_method_type == "triangle":
            nodes_to_consider = Regression.prune_with_triangle(
                self.data,
                self.vars,
                self.treatment,
                self.outcome,
                self.work_dir,
                top_n=self.cp.n,
                force=self.cp.force_triangle,
            )
        else:
            raise Exception(f"Invalid prune type: {self.cp.var_pruning_method_type}")
        return list(set(nodes_to_consider + [self.treatment, self.outcome]))

    def _find_edges_to_consider(
        self,
        nodes_to_consider: list[str],
    ) -> list[Types.Edge]:
        """
        Find the edges to consider when enumerating DAGs under the clustering method.

        Parameters:
            nodes_to_consider: The nodes to consider.

        Returns:
            A list of edges to consider.
        """

        # Enumerate all possible edges between nodes
        edges_to_consider = list(combinations(nodes_to_consider, 2))
        edges_to_consider = [
            edge
            for edge in edges_to_consider
            if not PreparedVariableName.same_base_var(edge[0], edge[1])
            and not (  # remove treatment-outcome edges of any kind
                PreparedVariableName.same_base_var(edge[0], self.treatment)
                and PreparedVariableName.same_base_var(edge[1], self.outcome)
            )
            and not (
                PreparedVariableName.same_base_var(edge[0], self.outcome)
                and PreparedVariableName.same_base_var(edge[1], self.treatment)
            )
        ]

        # If partial dag is given, remove its edges (and their reverses) from consideration.
        if self.true_graph is not None:
            edges_to_consider = [
                edge
                for edge in edges_to_consider
                if (edge not in self.true_graph.edges)
                and (edge[::-1] not in self.true_graph.edges)
            ]

        return edges_to_consider

    def _find_dags_to_consider(
        self,
        edges_to_consider: list[Types.Edge],
    ) -> list[nx.DiGraph]:
        """
        Find the DAGs to consider when enumerating DAGs under the clustering method.

        Parameters:
            edges_to_consider: The edges to consider.

        Returns:
            A list of DAGs to consider.
        """

        dags = []

        # Enumerate all possible DAGs based on edge presence.
        n = len(edges_to_consider)
        k = self.cp.num_edges

        # Initialize a list to store valid sequences
        valid_sequences = []
        stack = [([], 0, 0)]  # (sequence, non_none_count, index)

        while stack:
            sequence, count, index = stack.pop()
            # If the sequence is complete, add it to valid_sequences
            if index == n:
                valid_sequences.append(tuple(sequence))
                continue
            stack.append((sequence + [None], count, index + 1))
            if count < k:
                stack.append((sequence + [-1], count + 1, index + 1))
            if count < k:
                stack.append((sequence + [1], count + 1, index + 1))

        for sequence in valid_sequences:
            G = nx.DiGraph()
            edges = [
                edge[::edge_dir]
                for edge, edge_dir in zip(edges_to_consider, sequence)
                if edge_dir
            ]
            # Add partial dag in
            if self.true_graph is not None:
                edges.extend(self.true_graph.edges)

            edges.append((self.treatment, self.outcome))
            G.add_edges_from(edges)

            if nx.is_directed_acyclic_graph(G):
                dags.append(G)

        Printer.printv(f"Found {len(dags)} potential DAGs")
        return dags

    def _calculate_ates(
        self,
        dags: list[nx.DiGraph],
    ) -> dict[nx.DiGraph, tuple[float, float]]:
        """
        Calculate the ATEs of `treatment` on `outcome` for all DAGs in `dags`.

        Parameters:
            dags: The DAGs to consider.

        Returns:
            A dictionary mapping DAGs to their ATEs and P-values.
        """

        ates = {}
        for dag in tqdm(dags, "Processing DAGs"):
            results = ATECalculator.get_ate_and_confidence(
                self.data,
                self.vars,
                self.treatment,
                self.outcome,
                graph=dag,
                calculate_std_error=False,
            )
            ates[dag] = (results["ATE"], results["P-value"])
        return ates

    def _prune_aggregates(self, vars: list[str]) -> list[str]:
        """
        Prune aggregates by comparing average max abs difference between aggregates,
        and allowing for those in the bottom 10% compared to other variables
        to be represented by a single aggregate.

        Parameters:
            vars: The list of variables to prune.

        Returns:
            The pruned list of variables.
        """

        print("Starting aggregate pruning")
        Printer.set_warnings_to("ignore")
        base_vars = [PreparedVariableName(var).base_var() for var in vars]

        # Calculate the mean of the max abs difference for each base variable
        mean_diffs = {
            base_var: np.mean(
                self.data[
                    [
                        column
                        for column in self.sawmill.prepared_variable_names_with_base_x_and_no_pre_post_agg(
                            x=base_var
                        )
                    ]
                ].apply(lambda row: abs(row.max() - row.min()), axis=1)
            )
            for base_var in base_vars
        }
        mean_diffs = {key: val for key, val in mean_diffs.items() if not np.isnan(val)}

        # For each base variable, calculate the mean of the absolute values of the variables
        # with that base variable
        means = {
            base_var: np.mean(
                np.abs(
                    self.sawmill._prepared_log[
                        [
                            column
                            for column in self.sawmill.prepared_variable_names_with_base_x_and_no_pre_post_agg(
                                x=base_var
                            )
                        ]
                    ].values
                )
            )
            for base_var in base_vars
        }

        # normalize
        mean_diffs = {key: val / means[key] for key, val in mean_diffs.items()}
        mean_list = [mean for mean in mean_diffs.values() if mean != 0]
        cutoff = np.percentile(mean_list, 10) if mean_list else 0
        Printer.set_warnings_to("default")

        # Identify nodes based on the cutoff
        to_keep = []
        seen_base_vars = set()
        for var in vars:
            base_var = PreparedVariableName(var).base_var()

            if base_var != "Timestamp":
                if base_var not in mean_diffs or mean_diffs[base_var] > cutoff:
                    to_keep.append(var)
                elif base_var not in seen_base_vars:
                    print(f"- Using {var} as only aggregate for {base_var}")
                    to_keep.append(var)
                    seen_base_vars.add(base_var)

        print(f"Done pruning aggregates")
        return to_keep

    def _cluster(
        self,
    ) -> DendrogramRenderer:
        """
        Hierarchical Clustering using "ward" linkage method, measuring the distance
        between clusters based on the sum of squares within each cluster. As a result,
        it forms a hierarchy of clusters where data points are initially treated as individual
        clusters and then merged based on their similarity.

        Determine Optimal Clusters by calculate inconsistencies, which represent how dissimilar
        the merged clusters are. By comparing inconsistencies, you determine a threshold that helps
        you identify the optimal number of clusters. This threshold is based on a threshold multiplier
        (e.g., 1.0 times the maximum inconsistency)

        Returns:
            A DendrogramRenderer containing the dendrogram.
        """

        # Extract ATEs from effects and create linkage matrix
        ates = [val[0] for val in self.effects.values()]
        data = np.array(ates).reshape(-1, 1)
        linked: np.ndarray = linkage(data, method="ward")

        # Perform clustering
        if not self.cp.num_clusters:
            # If no number of clusters is specified, find the optimal number
            distances = linked[:, 2]
            threshold_multiplier = 0.5
            optimal_clusters = fcluster(
                linked, t=threshold_multiplier * np.max(distances), criterion="distance"
            )
        else:
            optimal_clusters = fcluster(
                linked, t=self.cp.num_clusters, criterion="maxclust"
            )

        self.num_clusters = int(np.max(optimal_clusters))
        print(f"Found {self.num_clusters} clusters")

        # Save cluster mappings
        optimal_clusters = np.array(optimal_clusters - 1)  # make it 0-indexed
        cluster_mapping = {
            dag: cluster for dag, cluster in zip(self.effects.keys(), optimal_clusters)
        }
        self.cluster_data = {
            i: (np.mean(data[optimal_clusters == i]), len(data[optimal_clusters == i]))
            for i in range(self.num_clusters)
        }

        # Create the dendrogram with the number of clusters we found, using 'lastp' truncation
        R = dendrogram(
            linked,
            orientation="top",
            labels=None,
            show_leaf_counts=False,
            truncate_mode="lastp",
            p=self.num_clusters,
            leaf_rotation=90,
            no_plot=True,
        )

        label_mapping = {
            R["leaves"][i]: self.cluster_data[i] for i in range(len(R["leaves"]))
        }
        self.tree, _ = EdgeOccurrenceTree.build_tree(linked, R["leaves"])
        self.tree.assign_dags_to_nodes(cluster_mapping)

        return DendrogramRenderer(
            linked,
            self.num_clusters,
            lambda x: "Mean: {:.3f}\n Count: {}".format(
                label_mapping[x][0], label_mapping[x][1], no_plot=True
            ),
        )

    def _display_important_edges(self):
        """
        Displays the results of the edge analysis.

        """

        # display ATE histogram and dendrogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ates = [val[0] for val in self.effects.values()]
        ax1.hist(ates)
        ax1.set_xlabel("ATE")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of ATE Values")
        dendrogram(
            self.dendrogram.linkage,
            orientation="top",
            labels=None,
            distance_sort="ascending",
            show_leaf_counts=True,
            truncate_mode="lastp",
            p=self.dendrogram.num_cluster,
            leaf_label_func=self.dendrogram.llf,
            ax=ax2,
            leaf_font_size=6,
        )
        ax2.set_title("Hierarchical clustering of ATE")

        # display edge frequency for each cluster
        fig, axes = plt.subplots(1, len(self.edge_counts.keys()), figsize=(12, 5))
        sorted_edge_keys = sorted(
            self.edge_counts[0].keys(), key=lambda k: self.edge_counts[0][k]
        )
        named_edge_keys = [
            (
                TagUtils.get_tag(self.vars, edge1, "prepared"),
                TagUtils.get_tag(self.vars, edge2, "prepared"),
            )
            for (edge1, edge2) in sorted_edge_keys
        ]

        for i, ax in enumerate(axes):
            edge_counts = {key: self.edge_counts[i][key] for key in sorted_edge_keys}
            ax.bar(range(len(sorted_edge_keys)), edge_counts.values())
            ax.set_xticks(range(len(sorted_edge_keys)))
            ax.set_xticklabels(named_edge_keys, rotation=90)
            ax.tick_params(axis="x", labelsize=8)
            ax.set_xlabel("Edges")
            ax.set_ylabel("Frequency")
            ax.set_title(
                "Edge counts for cluster centered at {:.3f}".format(
                    self.cluster_data[i][0]
                )
            )
        fig.tight_layout()

        for cluster_id, outlier in self.outliers.items():
            ate = self.cluster_data[cluster_id][0]
            df = pd.DataFrame.from_dict(
                {
                    "Edge": [
                        (
                            TagUtils.get_tag(self.vars, edge[0], "prepared"),
                            TagUtils.get_tag(self.vars, edge[1], "prepared"),
                        )
                        for edge in outlier.keys()
                    ],
                    "% Expectancy": list(outlier.values()),
                }
            )
            df["Status"] = np.where(df["% Expectancy"] > 0, "EXISTS", "DOES NOT EXIST")
            if self.cp.top_n:
                df_top = (
                    df[df["% Expectancy"] > 0]
                    .nlargest(self.cp.top_n, "% Expectancy")
                    .sort_values(by="% Expectancy", ascending=False)
                )
                df_bottom = (
                    df[df["% Expectancy"] < 0]
                    .nsmallest(self.cp.top_n, "% Expectancy")
                    .sort_values(by="% Expectancy")
                )
                df = pd.concat([df_top, df_bottom])
            else:
                df_top = df[df["% Expectancy"] > 0].sort_values(
                    by="% Expectancy", ascending=False
                )
                df_bottom = df[df["% Expectancy"] < 0].sort_values(by="% Expectancy")
                df = pd.concat([df_top, df_bottom])
            print(
                f"For ate = {ate}, the following edges are key assumptions made of the causal graph."
            )
            display(df)

    def _score_edges(self, graph, display_df=True, top_n=None):
        """
        TODO how do we want this for the paper?! Score the top_n edges we found
        according to the true graph by ___
        """

        # convert everything
        sawmill_graph = nx.DiGraph(
            [
                (
                    TagUtils.name_of(self.vars, edge[0], "prepared"),
                    TagUtils.name_of(self.vars, edge[1], "prepared"),
                )
                for edge in graph.edges
            ]
        )
        treatment = TagUtils.name_of(self.vars, self.treatment, "prepared")
        outcome = TagUtils.name_of(self.vars, self.outcome, "prepared")

        # identify backdoor nodes
        influential_nodes = set()
        ate_and_confidence = ATECalculator.get_ate_and_confidence(
            self.data,
            self.vars,
            treatment=treatment,
            outcome=outcome,
            graph=sawmill_graph,
            calculate_std_error=False,
            get_estimand=True,
        )
        ate = ate_and_confidence["ATE"]
        identified_estimand = ate_and_confidence["Estimand"]
        for _, vars in identified_estimand.backdoor_variables.items():
            influential_nodes = influential_nodes.union(vars)
        # identify backdoor edge
        scored_edges = []
        for edge in sawmill_graph.edges:
            if edge[0] in influential_nodes and (
                nx.has_path(sawmill_graph, edge[1], treatment)
                or nx.has_path(sawmill_graph, edge[1], outcome)
            ):
                scored_edges.append(edge)

        if top_n is None:
            top_n = len(scored_edges)

        # build up user graph according to outlier suggestions
        user_graph = nx.DiGraph([(treatment, outcome)])
        max_outliers = {}  # maps outliers to the max abs percent difference
        for _, outliers in self.outliers.items():
            for outlier in outliers:
                max_outliers[outlier] = max(
                    max_outliers.get(outlier, 0), np.abs(outliers[outlier])
                )
        top_outliers = list(outliers.keys())
        top_outliers.sort(key=lambda x: max_outliers[x], reverse=True)
        top_outliers = top_outliers[:top_n]
        user_graph.add_edges_from(
            [edge for edge in top_outliers if edge in sawmill_graph.edges]
        )

        found_edges = [
            1 if scored_edges[i] in user_graph.edges else 0
            for i in range(len(scored_edges))
        ]
        if display_df:
            df = pd.DataFrame.from_dict(
                {
                    "Edge": [
                        (
                            TagUtils.tag_of(self.data, edge[0], "prepared"),
                            TagUtils.tag_of(self.data, edge[1], "prepared"),
                        )
                        for edge in scored_edges
                    ],
                    "Score": found_edges,
                }
            ).sort_values(by="Score", ascending=False)
            display(df)
        edge_score = sum(found_edges) / len(found_edges)

        # ate score
        user_ate = ATECalculator.get_ate_and_confidence(
            self.data,
            self.vars,
            treatment=treatment,
            outcome=outcome,
            graph=user_graph,
            calculate_std_error=False,
        )["ATE"]
        ate_score = np.abs(ate - user_ate)
        return edge_score, ate_score


class ECCSATEChallenger:
    """
    A class to calculate edge changes impactful to an ATE calculation using ECCS.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        vars: pd.DataFrame,
        true_graph: Optional[nx.DiGraph],
        treatment: str,
        outcome: str,
        num_outputs: int = 10,
        method: str = "eccs-adjsetedit",
    ) -> None:
        """
        Initializes an ECCSATEChallenger.

        Parameters:
            data: The dataframe containing the data.
            vars: The dataframe containing information about the variables.
            true_graph: The starting graph to be used for causal analysis.
            treatment: The name or tag of the treatment variable.
            outcome: The name or tag of the outcome variable.
            num_outputs: The maximum number of candidate changes to output.
            method: The method to use for ATE calculation. Can be one of: "eccs-singleedit", "eccs-heuristicedit" or "eccs-adjsetedit".
        """
        if method not in ["eccs-singleedit", "eccs-heuristicedit", "eccs-adjsetedit"]:
            raise ValueError(f"Unknown method: {method}")

        self.data = data
        self.vars = vars
        self.true_graph = true_graph
        self.treatment = TagUtils.name_of(self.vars, treatment, "prepared")
        self.outcome = TagUtils.name_of(self.vars, outcome, "prepared")
        self.num_outputs = num_outputs

        if method == "eccs-singleedit":
            self.method = "best_single_edge_change"
        elif method == "eccs-heuristicedit":
            self.method = "astar_single_edge_change"
        else:
            self.method = "best_single_adjustment_set_change"
        self.eccs = ECCS(data, true_graph)
        self.eccs.set_treatment(self.treatment)
        self.eccs.set_outcome(self.outcome)

    def challenge(self) -> pd.DataFrame:
        """
        Invoke ECCS to produce a ranked list of edge changes that would most impact the ATE,
        and return the most impactful ones up to the number of outputs specified in the
        constructor.

        Returns:
            A dataframe containing the edge changes that would most impact the ATE.
        """

        # TODO: Bridge interpretation of the things ECCS returns to what this expects - 
        # e.g. that the outputs of adjsetedit are a set to be considered all together. 
        # Also, maybe provide support within Sawmill for fixed/banned etc. variables.

        edits, ate, _ = self.eccs.suggest(method=self.method)
        baseline_ate = ATECalculator.get_ate_and_confidence(
            self.data, self.vars, self.treatment, self.outcome, graph=self.true_graph
        )["ATE"]
        ate_ratio = max(abs(ate / baseline_ate), abs(baseline_ate / ate))
        graph_stats = []
        for edit in edits:
            d = {}
            d["Source"] = edit.src
            d["Source Tag"] = TagUtils.get_tag(self.vars, d["Source"], "prepared")
            d["Target"] = edit.dst
            d["Target Tag"] = TagUtils.get_tag(self.vars, d["Target"], "prepared")
            d["Change"] = str(edit.edit_type)
            d["ATE"] = ate
            d["Baseline ATE"] = baseline_ate
            d["ATE Ratio"] = ate_ratio

            graph_stats.append(d.copy())

        graphs_df = pd.DataFrame(graph_stats)
        graphs_df.sort_values(by="ATE Ratio", ascending=False, inplace=True)

        column_order = [
            "Source",
            "Source Tag",
            "Target",
            "Target Tag",
            "Change",
            "ATE",
            "Baseline ATE",
            "ATE Ratio",
        ]

        return graphs_df[column_order].head(self.num_outputs)
