#!/usr/bin/env python3

import csv
import json

from pathlib import Path
from pprint import pprint


def load_election(data_path):
    filepath = Path(data_path)
    basedir = filepath.parents[0]

    with open(filepath) as f:
        election = json.loads(f.read())

    data_filename = basedir / election['data']
    with open(data_filename) as f:
        reader = csv.DictReader(f, delimiter=',')
        election['data'] = list(reader)

    return election


if __name__ == '__main__':
    election = load_election('ballots/de2013.json')
    pprint(election['census'])

    minimum_seats = 1
    maximum_seats = election['seats']
    K = election['divisor']

    census_seats = {state: [0, 0.0] for state in election['census']}

    for state in election['census']:
        pass

    total_pop = sum(election['census'].values())
    print(total_pop)
