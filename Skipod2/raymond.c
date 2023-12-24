#include <mpi.h>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#define MARKER 1
#define REQUEST 0

#define QUEUE_SIZE 100

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int parent = (rank - 1) / 2;
    int children_finished = 0;

    int marker_direction = parent;

    int queue[QUEUE_SIZE];

    int queue_start = 0, queue_end = 0;

    char msg_type = REQUEST;
    bool has_marker = !rank;
    
    MPI_Status status;

    queue[queue_end++] = rank;

    MPI_Send(&msg_type, 1, MPI_CHAR, marker_direction, 0, MPI_COMM_WORLD);
    while (1) {
        MPI_Recv(&msg_type, 1, MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        if (msg_type == MARKER) {
            has_marker = true;
            marker_direction = queue[queue_start++];
            if (marker_direction == rank) {
                break;
            } else {
                MPI_Send(&msg_type, 1, MPI_CHAR, marker_direction, 0, MPI_COMM_WORLD);
                has_marker = false;
            }
        } else if (msg_type == REQUEST) {
            queue[queue_end++] = status.MPI_SOURCE;
            if (has_marker) {
                marker_direction = queue[queue_start++];
                msg_type = MARKER;
                MPI_Send(&msg_type, 1, MPI_CHAR, marker_direction, 0, MPI_COMM_WORLD);
                has_marker = false;
            } else {
                MPI_Send(&msg_type, 1, MPI_CHAR, marker_direction, 0, MPI_COMM_WORLD);
            }
        }
    }
    
    printf("%d aqcuared the marker from %d, entering critical section.\n", rank, status.MPI_SOURCE);
    fflush(stdout);
    const char* file_path = "critical.txt";
    FILE * file = fopen(file_path, "r");
    if (file) {
        printf("Failed to open file!\n");
        MPI_Finalize();
        return 1;
    } else {
        srand(time(NULL));
        unsigned int sleeptime = 1 + rand() % 10;
        sleep(sleeptime);
        printf("%d exiting critical section\n", rank);
        fflush(stdout);
        remove(file_path);
    }

    while (1) {
        MPI_Recv(&msg_type, 1, MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        if (msg_type == MARKER) {
            has_marker = true;
            marker_direction = queue[queue_start++];
            MPI_Send(&msg_type, 1, MPI_CHAR, marker_direction, 0, MPI_COMM_WORLD);
            has_marker = false;
        } else if (msg_type == REQUEST) {
            queue[queue_end++] = status.MPI_SOURCE;
            if (has_marker) {
                marker_direction = queue[queue_start++];
                msg_type = MARKER;
                MPI_Send(&msg_type, 1, MPI_CHAR, marker_direction, 0, MPI_COMM_WORLD);
                has_marker = false;
            } else {
                MPI_Send(&msg_type, 1, MPI_CHAR, marker_direction, 0, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Finalize();
    return 0;
}