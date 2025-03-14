"""
This script will go through all the `kernel-*.traceg` files in the given directory,
and calculate the total number of instructions executed that match the given pattern, stratified by the name of the kernel.

Usage: ./instruction_count.py <directory> <pattern>

Example: ./instruction_count.py ./traces "LDG"
This will calculate the total number of instructions executed that match the pattern "LDG" in all the `kernel-*.traceg` files in the directory `/path/to/directory/`

The output will be in the form:
| <pattern> count | kernel_name |
|-----------------|-------------|
| 123456          | kernel_name |
| 123456          | kernel_name |
...

the output will be in arbitrary order
"""

import os
import sys
from collections import defaultdict
import pandas as pd


def count_instructions(directory: str, patterns: list[str]) -> pd.DataFrame:
    # first is the number of times we've seen that kernel
    # the is the aggregate count of instructions of each pattern for each kernel
    df = pd.DataFrame(columns=["kernel_name"] + patterns)
    instruction_counts = defaultdict(lambda: [0] + [0] * len(patterns))

    # Iterate over all files in the given directory
    for filename in os.listdir(directory):
        if filename.startswith("kernel-") and filename.endswith(".traceg"):
            # print(f"Processing {filename}")
            file_path: str = os.path.join(directory, filename)

            # the kernel name is in the first line of the file,
            # all trace files start with the same basic structure
            with open(file_path, "r") as file:
                lines = file.readlines().__iter__()
                # the first line is of the form `-kernel name = <kernel_name>`
                kernel_name = next(lines).strip().split()[-1]

                # increment the count for this kernel
                instruction_counts[kernel_name][0] += 1

                # now we skip to the start of the actual instruction traces

                while ((line := next(lines, None)) is not None) and not line.startswith(
                    "insts = "
                ):
                    ...

                # now we are at the start of the instruction traces

                # traces format = [line_num] PC mask dest_num [reg_dests] opcode src_num [reg_srcs] mem_width [adrrescompress?] [mem_addresses] [immediate]

                while (line := next(lines, None)) is not None:
                    for i, pattern in enumerate(patterns):
                        if pattern in line:
                            instruction_counts[kernel_name][i + 1] += 1

    # turn our instruction counts into average counts
    df = pd.DataFrame.from_dict(
        instruction_counts,
        orient="index",
        columns=["count"] + ["avg. # of " + pattern for pattern in patterns],
    )
    df.reset_index(inplace=True)
    df.rename(columns={"index": "kernel_name"}, inplace=True)
    df.set_index("kernel_name", inplace=True)
    # calculate the average count for each kernel
    df = df.div(df["count"], axis=0)
    df.drop(columns=["count"], inplace=True)

    return df


PRINT_CSV = True


def print_instruction_counts(instruction_counts: pd.DataFrame):
    # Convert the dictionary to a DataFrame for better formatting
    df = instruction_counts.reset_index()

    # # reorder so that the kernel name is last
    # df = df.melt(id_vars=["kernel_name"], var_name="pattern", value_name="count")
    # df = df.pivot(index="kernel_name", columns="pattern", values="count").reset_index()
    # df.columns.name = None  # Remove the columns name
    # df = df.rename_axis(None, axis=1)  # Remove the index name
    # df = df.fillna(0)  # Fill NaN values with 0
    df = df.sort_values(by="kernel_name")  # Sort by kernel name

    # Print the DataFrame as a csv table
    if PRINT_CSV:
        print(df.to_csv(index=False, sep=",", header=True))
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./instruction_count.py <directory> <pattern>")
        sys.exit(1)

    directory = sys.argv[1]
    patterns = sys.argv[2:]

    instruction_counts = count_instructions(directory, patterns)
    print_instruction_counts(instruction_counts)
