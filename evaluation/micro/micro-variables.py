import sys
import pandas as pd

sys.path.append("../../")
from src.sawmill.sawmill import Sawmill
from evaluation.micro.micro_logs import gen_log
from datetime import datetime
import os


def main():
    variable_min = 1
    variable_max = 10
    variables = [10*i for i in range(variable_min, variable_max + 1)]

    f = open("micro-variables_runlog.txt", "w+")
    fr1 = open("micro-variables_resultslog.txt", "w+")
    fr1.write("variables, Parse Time, Prep Time\n")
    sys.stdout = f

    for v in variables:
        # Generate log
        L = 10000
        S = 10
        V = v
        C = 100-v
        filename = f"../../datasets_raw/micro-variables/log_{v}.log"
        gen_log(L, S, V, C, filename)
        print(f"Generated log with {v} variables")

        # Analyze log
        workdir = f"../../datasets/micro-variables/{datetime.now()}/"
        if not os.path.exists(workdir):
            os.makedirs(workdir)
        s = Sawmill(filename, workdir=workdir, skip_writeout=True)
        parse_time = s.parse(
            regex_dict={"LineID": r"line_\d+"}, sim_thresh=((C + 2) / 102), force=True
        )
        print(f"Shape of parsed log: {s.parsed_log.shape}")
        s.set_causal_unit("LineID")
        d = {k: "zero_imp" for k in s.parsed_log.columns[2:]}
        prep_time = s.prepare(
            custom_agg={"LineID": ["mode"]},
            custom_imp=d,
            ignore_uninteresting=False,
            force=True,
            drop_bad_aggs=False,
            reject_prunable_edges=False
        )
        print(f"Shape of prepared log: {s.prepared_log.shape}")
        s.prepared_log.head(10)

        fr1.write(f"{v},{parse_time},{prep_time}\n")
        fr1.flush()
        f.flush()

    f.close()
    fr1.close()


if __name__ == "__main__":
    main()
