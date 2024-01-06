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
    int sended = 0;

    if (rank > size - (size + 1) / 2 - 1) {
        sended = 2;
    } else if (rank == (size / 2 - 1) && !(size % 2)) {
        sended = 1;
    }

    int marker_direction = parent;

    int queue[QUEUE_SIZE];

    int queue_start = 0, queue_end = 0;

    char msg_type = !rank;
    bool has_marker = false;
    bool requested = true;

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
            }
            if (queue_start < queue_end) {
                msg_type = REQUEST;
                MPI_Send(&msg_type, 1, MPI_CHAR, marker_direction, 0, MPI_COMM_WORLD);
            }
            msg_type = MARKER;
            MPI_Send(&msg_type, 1, MPI_CHAR, marker_direction, 0, MPI_COMM_WORLD);
            has_marker = false;
        } else if (msg_type == REQUEST) {
            queue[queue_end++] = status.MPI_SOURCE;
        }
    }
    
    printf("%d acquared the marker from %d, entering critical section.\n", rank, status.MPI_SOURCE);
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
    for (; queue_start < queue_end || sended < 2;) {
        if (queue_start < queue_end && has_marker) {
            marker_direction = queue[queue_start++];
            if (queue_start < queue_end || sended < 1) {
                msg_type = REQUEST;
                MPI_Send(&msg_type, 1, MPI_CHAR, marker_direction, 0, MPI_COMM_WORLD);
            }
            msg_type = MARKER;
            MPI_Send(&msg_type, 1, MPI_CHAR, marker_direction, 0, MPI_COMM_WORLD);
            sended++;
            has_marker = false;
            if (queue_start < queue_end || sended < 1) {
                MPI_Recv(&msg_type, 1, MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
                if (msg_type == REQUEST) {
                    queue[queue_end++] = status.MPI_SOURCE;
                } else {
                    has_marker = true;
                }
            }
        } else if (has_marker) {
            MPI_Recv(&msg_type, 1, MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            queue[queue_end++] = status.MPI_SOURCE; 
        } else {
            MPI_Recv(&msg_type, 1, MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            if (msg_type == REQUEST) {
                queue[queue_end++] = status.MPI_SOURCE;
            } else {
                has_marker = true;
            }
        }
    }
    printf("%d finished\n", rank);
    MPI_Finalize();
    return 0;
}