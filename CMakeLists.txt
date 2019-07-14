cmake_minimum_required(VERSION 3.8)
project(mpiq)

include(blt/SetupBLT.cmake)

blt_add_executable( NAME mpiq
		    SOURCES mpiq.c
		    DEPENDS_ON cuda mpi hwloc)

