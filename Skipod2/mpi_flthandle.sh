#!/bin/bash -eu
mpicc redb_3d_mpi_flthandle.c -o redb_3d_mpi_flthandle
mpiexec -np 4 --with-ft ulfm ./redb_3d_mpi_flthandle