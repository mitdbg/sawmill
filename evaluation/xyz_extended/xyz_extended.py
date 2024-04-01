import sys
import pandas as pd

sys.path.append("../..")
from src.sawmill.sawmill import Sawmill
from evaluation.xyz_extended.xyz_extended_gen import xyz_extended_gen, DEFAULT_ARGS
import datetime
import os


def main():
    num_total_variables_options = [10,100,1000]
    noise_radius_options = [1, 5, 10]

    ts = datetime.datetime.utcnow()
    f = open(f"xyz_extended_runlog_{ts}.txt", "a+")
    fr = open(f"xyz_extended_resultslog_{ts}.txt", "a+")
    fr.write("V,R,recovered_ATE,original_ATE,error_pct,parse_time,prep_time,model_time,relative_time,user_interactions,mean_irrelevant_cands\n")
    sys.stdout = f

    
    #Run this for each file with suffix .log in the directory dir 
    for i in os.listdir('~/causal-log/datasets_raw/xyz_extended'):
        if i.endswith(".log"):
            filename = '~/causal-log/datasets_raw/xyz_extended/' + i

            with open(filename.split('.')[0] + ".json", "r") as fjson:
                l = fjson.readlines()
                num_total_variables = int(l[3].split(':')[1].strip().strip(','))
                noise_radius = int(l[4].split(':')[1].strip())
    #for num_total_variables in num_total_variables_options:
    #    for noise_radius in noise_radius_options:
    #        # Generate the log file
    #        args = DEFAULT_ARGS.copy()
    #        args["num_total_variables"] = num_total_variables
    #        args["noise_radius"] = noise_radius
    #        filename = xyz_extended_gen(args)

            # Parse and prepare the log file
            s = Sawmill(
                filename=filename,
                workdir="../../datasets/xyz_extended/",
            )
            parse_time = s.parse(
                regex_dict={
                    "timestamp": r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z",
                    "machine": r"machine_\d+",
                }
                
            )
            s.set_causal_unit("machine")
            prep_time = s.prepare(count_occurences=True, reject_prunable_edges=False)

            # Print statistics as a sanity check
            print("----------------------------------------")
            print("num_total_variables:", num_total_variables)
            print("noise_radius:", noise_radius)
            print("number of parsed templates:", len(s.parsed_templates))
            print("number of parsed variables:", len(s.parsed_variables))
            print("number of prepared variables:", len(s.prepared_variables))

            # Determine ranks of candidate causes
            c1, cand_time_1 = s.explore_candidate_causes("y mean")
            print(c1)
            top3_y_cands = c1["Candidate Tag"].tolist()[:3]

            x_to_y_rank = c1.index[c1["Candidate Tag"] == "x mean"].tolist()
            if len(x_to_y_rank) == 0:
                x_to_y_rank = -1
            else:
                x_to_y_rank = x_to_y_rank[0]
            z_to_y_rank = c1.index[c1["Candidate Tag"] == "z mean"].tolist()
            if len(z_to_y_rank) == 0:
                z_to_y_rank = -1
            else:
                z_to_y_rank = z_to_y_rank[0]

            s.accept("x mean", "y mean")
            s.accept("z mean", "y mean")

            next_exploration = s.suggest_next_exploration()
            print(f"next exploration: {next_exploration}")

            c2, cand_time_2 = s.explore_candidate_causes("x mean", lasso_alpha=0.1)
            print(c2)
            top3_x_cands = c2["Candidate Tag"].tolist()[:3]

            z_to_x_rank = c2.index[c2["Candidate Tag"] == "z mean"].tolist()
            if len(z_to_x_rank) == 0:
                z_to_x_rank = -1
            else:
                z_to_x_rank = z_to_x_rank[0]

            # Create the graph and calculate ATEs
            s.accept("z mean", "x mean")
            unadjusted = s.get_unadjusted_ate("x mean", "y mean")
            adjusted = s.get_adjusted_ate("x mean", "y mean")
            print("unadjusted:", unadjusted)
            print("adjusted:", adjusted)

            #print(s.challenge_ate('x mean', 'y mean'))

            # Write results to file
            error_pct = abs(adjusted-2.0)/2*100.0
            total_cand_time = float(cand_time_1) + float(cand_time_2)
            fr.write(f"{num_total_variables},{noise_radius},{adjusted:.2f},{2:.2f},{error_pct:.2f},{float(parse_time):.2f},{float(prep_time):.2f},{total_cand_time:.2f},N/A,9,0\n")

           
            fr.flush()

    f.close()
    fr.close()


if __name__ == "__main__":
    main()
