#!/usr/bin/env python3

import argparse
import textwrap

from pprint import pprint

import common
import german


run_election_types = {
    'Germany': (german.process_params, german.run_german_election),
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run various types of elections.')
    parser.add_argument('definition', help='Definition file for the election.')

    parser.add_argument('-p', '--param', nargs=2, action='append',
                        metavar=('NAME', 'VALUE'), help='Override parameters '
                        'for the apportionment method.')

    args = parser.parse_args()

    election = common.load_election(args.definition)

    try:
        process_params, run_election = run_election_types[election['apportionment']]

        provided_params = election.get('params', {})
        if args.param:
            provided_params.update(dict(args.param))

        params = process_params(provided_params)
        run_election(election, params)

    except KeyError:
        print(textwrap.dedent(f"""\
        Invalid apportionment method "{election['apportionment']}"

        Valid apportionment methods are: {', '.join(run_election_types.keys())}
        """).strip())
