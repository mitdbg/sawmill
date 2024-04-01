import sys
import pandas as pd

sys.path.append("../../")
from src.sawmill.sawmill import Sawmill
from evaluation.micro.micro_logs import gen_log
import datetime
import json


def main():
    length_exp_min = 1
    length_exp_max = 6
    lengths = [10**i for i in range(length_exp_min, length_exp_max + 1)]

    f = open("micro-length_runlog.txt", "w+")
    fr1 = open("micro-length_resultslog.txt", "w+")
    fr1.write("Length, Parse Time, Prep Time\n")
    sys.stdout = f

    for l in lengths:
        # Generate log
        L = l
        S = 10
        V = 1
        C = 10
        filename = f"../../datasets_raw/micro-length/log_{l}.log"
        gen_log(L, S, V, C, filename)
        print(f"Generated log of length {l}")

        # Analyze log
        workdir = f"../../datasets/micro-length/"
        s = Sawmill(filename, workdir=workdir, skip_writeout=True)
        parse_time = s.parse(
            regex_dict={"LineID": r"line_\d+"}, sim_thresh=(12 / 13), force=True
        )
        print(f"Number of templates: {len(s.parsed_templates)}")
        print(f"Number of parsed variables: {len(s.parsed_variables)}")
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

        fr1.write(f"{l},{parse_time},{prep_time}\n")
        fr1.flush()
        f.flush()

    f.close()
    fr1.close()


if __name__ == "__main__":
    main()
