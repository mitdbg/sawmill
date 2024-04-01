import sys
import pandas as pd

sys.path.append("../../")
from src.sawmill.sawmill import Sawmill
from evaluation.micro.micro_logs import gen_log
from datetime import datetime
import os


def main():
    template_exp_min = 1
    template_exp_max = 4
    templates = [10**i for i in range(template_exp_min, template_exp_max + 1)]

    f = open("micro-templates_runlog.txt", "w+")
    fr1 = open("micro-templates_resultslog.txt", "w+")
    fr1.write("Templates, Parse Time, Prep Time\n")
    sys.stdout = f

    for t in templates:
        # Generate log
        L = 10000
        S = t
        V = 1
        C = 10
        filename = f"../../datasets_raw/micro-templates/log_{t}.log"
        gen_log(L, S, V, C, filename)
        print(f"Generated log with {t} templates")

        # Analyze log
        workdir = f"../../datasets/micro-templates/{datetime.now()}/"
        if not os.path.exists(workdir):
            os.makedirs(workdir)
        s = Sawmill(filename, workdir=workdir, skip_writeout=True)
        parse_time = s.parse(
            regex_dict={"LineID": r"line_\d+"}, sim_thresh=12/13, force=True
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

        fr1.write(f"{t},{parse_time},{prep_time}\n")
        fr1.flush()
        f.flush()

    f.close()
    fr1.close()


if __name__ == "__main__":
    main()
