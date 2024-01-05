#!/bin/bash -eu
mpicc raymond.c -o raymond
mpiexec -np 25 --host=eugene:25 ./raymond