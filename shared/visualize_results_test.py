import argparse
import pytest


# fixture to create the arg parser
@pytest.fixture
def parser():
    parser = argparse.ArgumentParser(
        description="Create plots from the organized simulation results."
    )
    parser.add_argument(
        "--prune_outlier",
        "--prune-outlier",
        const=3.0,
        type=float,
        default=None,
        metavar="threshold",
        nargs="?",
        help="Prune outliers from the data based on the specified metric, you can optionally provide a threshold as well.",
    )
    return parser


prune_outliers_data = [
    (["--prune_outlier"], 3.0),
    ([], None),
    (["--prune-outlier", "5.0"], 5.0),
]


@pytest.mark.parametrize("args, expected", prune_outliers_data)
def test_prune_outlier_args(parser, args, expected):
    parsed_args = parser.parse_args(args)
    assert (
        parsed_args.prune_outlier == expected
    ), f"Expected {expected}, got {parsed_args.prune_outlier}"
