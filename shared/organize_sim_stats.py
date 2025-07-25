#!/usr/bin/env python3
#
# Given a stats.csv file generated by accel-sim (with get_stats.py), this script will:
# 1. Read the stats.csv file
# 2. Organize the data into a dataframe mapping jobs to the kernel names to the stats
# 3. Save the organized data into a new CSV files in a folder with the same name as the input file but with `organized_` prefix, the files will be named by the job name
#
# The complication here is the format of accelsim's CSV files.
#
# They are only ever `number of configs + 1` columns, and have multiple sections separated by
# "----------------------------------------------------------------------------------------------------,"
#
# each section is roughly of the format:
# ```
# ----------------------------------------------------------------------------------------------------,
# <stat>
# APPS,<first config><,...>
# <job name>--<kernel name>,<the result for that kernel>
# <job name>--<kernel name>,<the result for that kernel>
# <...>
#
# ```
#
# The first 2 sections, `Accel-Sim-build` and `GPGPU-Sim-build` are ignored.

from attr import dataclass
import pandas as pd
import os
import sys
import re
import argparse
import numpy as np
import csv


def extract_job_name(kernel: str) -> str:
    """
    Extract the job name from the kernel string.

    converting something like:
    mnist_mnist_dummy.py/__device_cuda___batch_size_1___max_iters_2--_ZN2at6native39_GLOBAL__N__5cd812b1_7_Loss_cu_5b0651e139nll_loss_backward_reduce_cuda_kernel_2dIflEEvPT_PKS3_PKT0_S6_S6_biill--1

    to:
    mnist_mnist_dummy.py
    """
    match = re.match(r"^(.*?)--", kernel)
    if match:
        return match.group(1)
    return ""


def extract_kernel_name(kernel: str) -> str:
    """
    Extract the kernel name from the kernel string.

    converting something like:
    mnist_mnist_dummy.py/__device_cuda___batch_size_1___max_iters_2--_ZN2at6native39_GLOBAL__N__5cd812b1_7_Loss_cu_5b0651e139nll_loss_backward_reduce_cuda_kernel_2dIflEEvPT_PKS3_PKT0_S6_S6_biill--1

    to:
    _ZN2at6native39_GLOBAL__N__5cd812b1_7_Loss_cu_5b0651e139nll_loss_backward_reduce_cuda_kernel_2dIflEEvPT_PKS3_PKT0_S6_S6_biill--1
    """
    parts = kernel.split("--")
    return "--".join([parts[-2], parts[-1]])


def job_stat_dict_to_dataframe(
    dict_df: dict[str, dict[str, float]], stats: list[str]
) -> pd.DataFrame:
    """
    Convert a nested dictionary of kernel stats to a pandas dataframe.
    """
    df = pd.DataFrame.from_dict(dict_df, orient="index", columns=stats, dtype=float)

    # name the index
    df.index.name = "Kernel Name"

    # we want to get the demangled kernel name and add it as a new column,
    # to do this we need to run the demangler on each kernel name

    # get the demangled kernel name
    args = " ".join(df.index.map(lambda kernel: kernel.split("--")[0]))
    demangled_kernel_names = os.popen(f"c++filt {args}").read().strip().split("\n")
    # add the demangled kernel name to the dataframe
    df["Kernel Name (Demangled)"] = demangled_kernel_names

    return df


@dataclass
class StatsParser:
    input_file: str
    output_dir: str

    dfs: dict[str, pd.DataFrame] = {}

    def parse(self) -> dict[str, pd.DataFrame]:
        """
        Parse the input CSV file and organize the data into a dataframe.
        """
        # this dataframe will be 3-Dimensional, and indexed like so:
        # df[job_name][kernel_name][stat]
        # where kernel_name is the name of the kernel, and stat is the name of the stat
        #
        # NOTE: this assumes that you're only running one config at a time
        #
        # the final value in each such index will be the value of the stat for that kernel and configuration
        dfs: dict[str, dict[str, dict[str, float]]] = {}

        with open(self.input_file, "r") as f:
            reader = csv.reader(f)
            # track which section we are in
            section = 0
            # get the number of configurations from the first section
            reader.__next__()
            reader.__next__()
            stats: list[str] = []
            # continue skipping lines until we finish the first 2 sections
            for row in reader:
                if len(row) == 0 or row[0].startswith("APPS"):
                    continue
                if row[0].startswith(
                    "----------------------------------------------------------------------------------------------------"
                ):
                    section += 1
                    if section == 2:
                        # we are now in the third section, which contains the kernel names and their results
                        break

            current_stat: str | None = None
            # Read the rest of the file
            for row in reader:
                if len(row) == 0 or row[0].startswith("APPS"):
                    continue
                if row[0].startswith(
                    "----------------------------------------------------------------------------------------------------"
                ):
                    section += 1
                    current_stat = None
                    continue

                if current_stat is None:
                    # this is the first line of a new section, which contains the stat name
                    current_stat = row[0].strip()
                    stats.append(current_stat)
                    continue

                job = extract_job_name(row[0])
                if job == "":
                    print(
                        f"Could not extract job name from kernel string: {row[0]}",
                        file=sys.stderr,
                    )
                    continue
                if job not in dfs:
                    dfs[job] = {}
                df = dfs[job]

                kernel = extract_kernel_name(row[0])
                # if the kernel name is empty, skip this row
                if kernel == "":
                    continue
                # if the kernel name is not in the dataframe, add it
                if kernel not in df:
                    df[kernel] = {}

                # add the value to the dataframe
                # convert the value to a float, if possible
                try:
                    value = float(row[1])
                except ValueError:
                    # if the value is not a float, skip it
                    print(
                        f"Value {row[1]} is not a float, skipping row: {row}",
                        file=sys.stderr,
                    )
                    df[kernel][current_stat] = np.nan
                    continue
                # add the value to the dataframe
                df[kernel][current_stat] = value

        # convert the dictionaries to dataframes
        if len(dfs) < 1:
            raise ValueError("No data found in input file.")
        self.dfs = {}
        for job, df in dfs.items():
            self.dfs[job] = job_stat_dict_to_dataframe(df, stats)

        return self.dfs

    @classmethod
    def from_args(cls) -> "StatsParser":
        """
        Create a StatsParser object from command line arguments.
        """
        parser = argparse.ArgumentParser(
            description="Organize accel-sim stats CSV file."
        )
        parser.add_argument(
            "input_file", type=str, help="Path to the input stats.csv file."
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            help="Path to of directory to output organized CSV file.",
        )

        args = parser.parse_args()
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file {args.input_file} does not exist.")
        if not os.path.isfile(args.input_file):
            raise ValueError(f"Input file {args.input_file} is not a file.")
        if args.output_dir and os.path.exists(args.output_dir):
            raise FileExistsError(f"Output directory {args.output_dir} already exists.")
        if args.output_dir and not os.path.isdir(args.output_dir):
            raise ValueError(f"Output file {args.output_dir} is not a file.")

        if args.output_dir is None:
            args.output_dir = os.path.join(
                os.path.dirname(args.input_file),
                # organized_<input_file_name without extension
                "organized_" + os.path.splitext(os.path.basename(args.input_file))[0],
            )
        return cls(args.input_file, args.output_dir)


def main() -> None:
    """
    Main function to parse the input CSV file and organize the data into a dataframe.
    """
    parser = StatsParser.from_args()
    dfs = parser.parse()

    # save the dataframes to CSV files
    os.makedirs(parser.output_dir, exist_ok=True)
    for job, df in dfs.items():
        # sanitize the job name to be a valid file name
        sanitized_job_name = re.sub(r"[^\w\-_\. ]", "_", job)
        output_file = os.path.join(parser.output_dir, f"{sanitized_job_name}.csv")
        df.to_csv(output_file, index=True, sep="\t", quoting=csv.QUOTE_NONNUMERIC)
        print(f"\n**Organized data for job {job} saved to {output_file}**\n")

        # print the unique denmangled kernel names
        unique_kernel_names = df["Kernel Name (Demangled)"].unique()
        print(f"\n{len(unique_kernel_names)} Unique kernel names:")
        print("\n".join(unique_kernel_names))


if __name__ == "__main__":
    main()
