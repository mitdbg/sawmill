import sys

sys.path.append("../..")
from src.sawmill.sawmill import Sawmill
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("expand_frame_repr", False)
pd.set_option("display.max_colwidth", None)

# Base
s = Sawmill(
    "../../datasets_raw/tpc-ds/parameter_sweep_1.log", workdir="../../datasets/tpc-ds"
)
s.parse(
    regex_dict={
        "Date": r"\d{4}-\d{2}-\d{2}",
        "Time": r"\d{2}:\d{2}:\d{2}\.\d{3}(?= EST \[ )",
        "sessionID": r"(?<=EST \[ )\S+\.\S+",
        "tID": r"3/\d+(?= ] )",
    },
    message_prefix=r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{3}",
)
s.include_in_template("d2cdc31b_13")
s.set_causal_unit("sessionID")
s.prepare(count_occurences=True, custom_agg={"sessionID": ["mode"]})

s.accept("work_mem mean", "duration max", interactive=False)
s.accept("seq_page_cost mean", "duration max", interactive=False)
s.accept("random_page_cost mean", "duration max", interactive=False)
s.accept("max_parallel_workers mean", "duration max", interactive=False)
s.accept("maintenance_work_mem mean", "duration max", interactive=False)
s.accept("effective_cache_size mean", "duration max", interactive=False)
s.reject_undecided_incoming("duration max")


old_prepared = s.prepared_log.copy()
allowed_fully = s._prepared_log[
    (
        (s._prepared_log["da4032f7_15+mean"] == 128)
        & (s._prepared_log["9d2f8f03_15+mean"] == 2)
    )
    | (
        (s._prepared_log["da4032f7_15+mean"] == 256)
        & (s.prepared_log["9d2f8f03_15+mean"] == 1)
    )
].copy()
allowed_half = s.prepared_log[
    (
        (s.prepared_log["da4032f7_15+mean"] == 128)
        & (s.prepared_log["9d2f8f03_15+mean"] == 1)
    )
    | (
        (s.prepared_log["da4032f7_15+mean"] == 256)
        & (s.prepared_log["9d2f8f03_15+mean"] == 2)
    )
].copy()
allowed_half.drop_duplicates(
    inplace=True,
    subset=[
        "da4032f7_15+mean",
        "df623689_15+mean",
        "97a1388a_15+mean",
        "9d2f8f03_15+mean",
        "9eea70b1_15+mean",
        "8d5769b2_15+mean",
    ],
)


s._prepared_log = pd.concat([allowed_fully, allowed_half])


s.clear_graph(clear_edge_states=False)



s.accept("work_mem mean", "duration max", interactive=False)
s.accept("seq_page_cost mean", "duration max", interactive=False)
s.accept("random_page_cost mean", "duration max", interactive=False)
s.accept("max_parallel_workers mean", "duration max", interactive=False)
s.accept("maintenance_work_mem mean", "duration max", interactive=False)
s.accept("effective_cache_size mean", "duration max", interactive=False)
s.reject_undecided_incoming("duration max")


s.accept("work_mem mean", "max_parallel_workers mean")



print("ATE of max parallel workers on max latency:")
print(
    f"Unadjusted: {s.get_unadjusted_ate('max_parallel_workers mean', 'duration max')}\nAdjusted: {s.get_adjusted_ate('max_parallel_workers mean', 'duration max')}"
)


print(s.challenge_ate("max_parallel_workers mean", "duration max"))
