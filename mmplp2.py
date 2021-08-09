#!/usr/bin/env python3

from pprint import pprint

import common
import german


if __name__ == '__main__':
    election = common.load_election('ballots/de2017.json')
    results = german.run_german_election(election)
    pprint(results)
