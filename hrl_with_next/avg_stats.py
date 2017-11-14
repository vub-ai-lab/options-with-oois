#!/usr/bin/python3
#
# This file is part of OOIs.
#
# Copyright 2017, Vrije Universiteit Brussel (http://vub.ac.be)
#
# OOIs is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OOIs is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function

import sys
import math

from future_builtins import zip

if len(sys.argv) < 3:
    print("Usage: %s <column> [file...]" % sys.argv[0])
    sys.exit(0)

# Open all files
col = int(sys.argv[1])
files = [open(f, 'r') for f in sys.argv[2:]]

# Read and average each files
N = float(len(files))
i = 0
running_mean = None
running_err = None
elems = [0.0] * len(files)

running_mean = None
running_err = 1.0

for elements in zip(*files):
    i += 1

    if (i % 64) != 0:
        continue

    for j in range(len(files)):
        elems[j] = float(elements[j].strip().split()[col])

        if elems[j] < -3.0:
            elems[j] = 0.0

    mean = sum(elems) / N
    var = sum([(e - mean)**2 for e in elems])
    std = math.sqrt(var)
    err = std / N

    if running_mean is None:
        running_mean = mean
    else:
        running_mean = 0.9 * running_mean + 0.1 * mean

    running_err = 0.9 * running_err + 0.1 * err

    print(i / 1000., running_mean, running_mean + running_err, running_mean - running_err)
