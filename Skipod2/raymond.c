#include <mpi.h>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int parent = (rank - 1) / 2;
    int left = 2 * rank + 1;
    int right = 2 * rank + 2;

    int tmp = 0;
    if (rank != 0) {
        printf("%d requested marker from %d.\n", rank, parent);
        MPI_Recv(&tmp, 1, MPI_INT, parent, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    printf("%d aqcuared the marker.\n", rank);

    const char* file_path = "critical.txt";
    FILE * file = fopen(file_path, "r");
    if (file) {
        printf("Failed to open file!\n");
        MPI_Finalize();
        return 1;
    } else {
        srand(time(NULL));
        unsigned int sleeptime = 1 + rand() % 10;
        printf("Process %d started working. It'll finish in %u second(s).\n", rank, sleeptime);
        sleep(sleeptime);
        printf("Process %d finished working.\n", rank);
        remove(file_path);
    }

    if (left < size) {
        int tmp_l = 1;
        printf("%d sending marker to %d.\n", rank, left);
        MPI_Send(&tmp_l, 1, MPI_INT, left, 0, MPI_COMM_WORLD);
        printf("%d requested marker from %d.\n", rank, left);
        MPI_Recv(&tmp_l, 1, MPI_INT, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("%d aqcuared the marker.\n", rank);
    }

    if (right < size) {
        printf("%d sending marker to %d.\n", rank, right);
        MPI_Send(&tmp, 1, MPI_INT, right, 0, MPI_COMM_WORLD);
        if (tmp) {
            printf("%d requested marker from %d.\n", rank, right);
            MPI_Recv(&tmp, 1, MPI_INT, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%d aqcuared the marker.\n", rank);
        }
    }

    if (rank != 0 && tmp) {
        printf("%d sending marker to %d.\n", rank, parent);
        MPI_Send(&tmp, 1, MPI_INT, parent, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}