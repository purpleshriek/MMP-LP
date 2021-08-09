#!/usr/bin/env python3

import argparse
import textwrap

from pprint import pprint

import common
import german


run_election_types = {
    'Germany': german.run_german_election
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run various types of elections.')
    parser.add_argument('definition', help='Definition file for the election.')

    args = parser.parse_args()

    election = common.load_election(args.definition)

    try:
        fun = run_election_types[election['apportionment']]
        results = fun(election)
        pprint(results)
    except KeyError:
        print(textwrap.dedent(f"""\
        Invalid apportionment method "{election['apportionment']}"

        Valid apportionment methods are: {', '.join(run_election_types.keys())}
        """).strip())
