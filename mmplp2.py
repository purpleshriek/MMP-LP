#!/usr/bin/env python3

import csv
import json
import time

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


def apportion_webster(census, seats):
    total_pop = sum(census.values())
    divisor = total_pop / seats

    while True:
        reps = {
            district: round(pop / divisor)
            for district, pop in census.items()
        }
        current_seats = sum(reps.values())
        if current_seats == seats: break

        error_factor = current_seats / seats
        divisor *= error_factor

    return reps


if __name__ == '__main__':
    election = load_election('ballots/de2013.json')

    minimum_seats = 1
    K = election['divisor']

    reps = apportion_webster(election['census'], election['seats'])

    pprint(reps)
