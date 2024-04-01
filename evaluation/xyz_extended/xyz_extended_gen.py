import argparse
import numpy as np
import datetime
import json

# z is baseline load
# x is x_factor * z + random noise between -noise_radius and noise_radius
# y is 2 * x_factor * z + random noise between -noise_radius and noise_radius


DEFAULT_ARGS = {
    "machines": 1000,
    "length_per_machine": 1000,
    "num_total_variables": 100,
    "noise_radius": 1,
}


def clip(x, args):
    return max(min(x, 100), 0)


def xyz_extended_gen(args):
    ts = datetime.datetime.utcnow()
    filename = f"~/causal-log/datasets_raw/xyz_extended/log_{ts.year:04d}-{ts.month:02d}-{ts.day:02d}_{ts.hour:02d}:{ts.minute:02d}:{ts.second:02d}"

    # Dump arguments
    with open(filename + ".json", "w+") as f:
        json.dump(args, f, indent=2)

    x_index = args["num_total_variables"] - 2
    y_index = args["num_total_variables"] - 1
    z_index = args["num_total_variables"]

    # Compose the log
    with open(filename + ".log", "a+") as f:
        for machine in range(args["machines"]):
            # Set this machine's z and l
            z = np.random.uniform(0, 10)
            x_base = np.random.uniform(0, 10)

            # Draw this machine's variables to report and ensure each variable is reported at least once.
            to_report = [i for i in range(1, args["num_total_variables"] + 1)]
            to_report.extend(
                np.random.randint(
                    1,
                    args["num_total_variables"] + 1,
                    args["length_per_machine"] - args["num_total_variables"],
                ).tolist()
            )

            for line in range(args["length_per_machine"]):
                machine_str = f"machine_{machine}"
                ts_str = f"{ts.year:04d}-{ts.month:02d}-{ts.day:02d}T{ts.hour:02d}:{ts.minute:02d}:{ts.second:02d}.{ts.microsecond:06d}Z"

                x = x_base + z + np.random.normal(0, args[
                        "noise_radius"
                    ])
                y = 2 * x + 3 * z + np.random.normal(0, args[
                        "noise_radius"
                    ])

                # Compute and print the variable
                if to_report[line] == x_index:
                    f.write(f"{ts_str} {machine_str} x={clip(x, args)}\n")
                elif to_report[line] == y_index:
                    f.write(f"{ts_str} {machine_str} y={clip(y, args)}\n")
                elif to_report[line] == z_index:
                    # Report z
                    f.write(f"{ts_str} {machine_str} z={z}\n")
                else:
                    # Report a random variable
                    value = np.random.random() * 100
                    f.write(f"{ts_str} {machine_str} var_{to_report[line]}={value}\n")

                # Update timestamp
                ts += datetime.timedelta(milliseconds=1)

    return filename + ".log"


if __name__ == "__main__":
    # Accept arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--machines", type=int, default=1000)
    parser.add_argument("--length_per_machine", type=int, default=1000)
    parser.add_argument("--num_total_variables", type=int, default=100)
    parser.add_argument("--noise_radius", type=float, default=1)
    args = parser.parse_args().__dict__
    xyz_extended_gen(args)
