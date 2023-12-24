#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <setjmp.h>
#include <time.h>
#include <unistd.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))

#define N (2*2*2*2*2*2 + 2)
#define TAG_PASS_FIRST 0xA
#define TAG_PASS_LAST 0xB
#define N2 (N * N)
#define N3 (N * N2)

jmp_buf jbuf;
char file_path[] = "checkpoints";
double maxeps = 0.1e-7;
int itmax = 101;
int i, j, k;
double w = 0.5;
double eps;
double b, s = 0.;
double ***A;
MPI_Comm global_comm = MPI_COMM_WORLD;
int rank_to_kill = -50;

// algorithm functions
void relax();
void init();
void verify();

// functions for communication
void pass_first_row();
void pass_last_row();
void wait_all();

// functions for data safety
void save_checkpoint();
void load_checkpoint();

int size, rank, fst_r, lst_r, cnt_r;

MPI_Request req_buf[4];
MPI_Status stat_buf[4];

static void error_handler(MPI_Comm *comm, int *err, ...) {
    int len;
    char errstr[MPI_MAX_ERROR_STRING];

    rank_to_kill = -200;

    MPIX_Comm_shrink(*comm, &global_comm);
    
    MPI_Comm_rank(global_comm, &rank);
    MPI_Comm_size(global_comm, &size);
    
    MPI_Error_string(*err, errstr, &len);
    printf("Rank %d / %d: Notified of error %s\n", rank, size, errstr);
    
    MPI_Barrier(global_comm);

    //adaptive choice of rows(depends on process count)
    fst_r = (N - 2) / size * rank + 1;
    lst_r = (N - 2) / size * (rank + 1) + 1;
    cnt_r = lst_r - fst_r;

    A = malloc((cnt_r + 2) * sizeof(*A));

    for (i = 0; i < cnt_r + 2; i++) {
        A[i] = malloc(N * sizeof(*A[i]));
        for (j = 0; j < N; j++)
            A[i][j] = malloc(N * sizeof(*A[i][j]));
    }

    longjmp(jbuf, 0);
}


int main(int argc, char **argv) {
    int it = 1;

    MPI_Init(&argc, &argv);

    MPI_Errhandler errh;

    MPI_Comm_rank(global_comm, &rank);
    MPI_Comm_size(global_comm, &size);

    MPI_Comm_create_errhandler(error_handler, &errh);
    MPI_Comm_set_errhandler(global_comm, errh);

    MPI_Barrier(global_comm);

    double time_start, time_end;
    time_start = MPI_Wtime();
    
    init();
    save_checkpoint();
    setjmp(jbuf);
    for (it; it <= itmax; it++) {
        
        if (it == itmax) {
            pass_last_row();
            pass_first_row();
            wait_all();
            verify();
            break;
        }

        eps = 0.;
        relax();
        //avoid output duplication
        if (!rank)
            printf("it=%4i   eps=%f\n", it, eps);
        if (eps < maxeps)
            break;
    }

    time_end = MPI_Wtime();

    if (!rank)
        printf("Elapsed time: %lf.\n", time_end - time_start);

    MPI_Finalize();

    return 0;
}

void init() {
    //adaptive choice of rows(depends on process count)
    fst_r = (N - 2) / size * rank + 1;
    lst_r = (N - 2) / size * (rank + 1) + 1;
    cnt_r = lst_r - fst_r;

    A = malloc((cnt_r + 2) * sizeof(*A));

    for (i = 0; i < cnt_r + 2; i++) {
        A[i] = malloc(N * sizeof(*A[i]));
        for (j = 0; j < N; j++) {
            A[i][j] = malloc(N * sizeof(*A[i][j]));
            if (j <= N - 1) {
                    for (k = 0; k < N ; k++) {
                    //row check is unnecessary now
                    if (j == 0 || j == N - 1 || k == 0 || k == N - 1)
                        A[i][j][k] = 0.;
                    else
                        A[i][j][k] = (4. + i + j + k);
                }
            }    
        }
    }
}

void relax() {    
    load_checkpoint();
    double eps_local = 0.;
    pass_last_row();
    pass_first_row();
    wait_all();
    for (i = 1; i < cnt_r + 1; i++)
        for (j = 1; j <= N - 2; j++)
            for (k = 1 + (i + j) % 2; k <= N - 2; k += 2) {
                b = w * ((A[i - 1][j][k] + A[i + 1][j][k] + A[i][j - 1][k] +
                          A[i][j + 1][k] + A[i][j][k - 1] + A[i][j][k + 1]) /
                          6. - A[i][j][k]);
                eps_local = Max(fabs(b), eps_local);
                A[i][j][k] = A[i][j][k] + b;
            }
    if (rank == rank_to_kill) {
        printf("Process %d. I guess I'll die...\n", rank);
        fflush(stdout);
        raise(SIGKILL);
    }
    rank_to_kill++;
    
    pass_last_row();
    pass_first_row();
    wait_all();
    for (i = 1; i < cnt_r + 1; i++)
        for (j = 1; j <= N - 2; j++)
            for (k = 1 + (i + j + 1) % 2; k <= N - 2; k += 2) {
                b = w * ((A[i - 1][j][k] + A[i + 1][j][k] + A[i][j - 1][k] +
                          A[i][j + 1][k] + A[i][j][k - 1] + A[i][j][k + 1]) /
                          6. - A[i][j][k]);
                A[i][j][k] = A[i][j][k] + b;
            }
    save_checkpoint();
    MPI_Allreduce(&eps_local, &eps, 1 , MPI_DOUBLE, MPI_MAX, global_comm);
}

void verify() {
    if (rank >= size)
        return;
    load_checkpoint();
    double s_local = 0.;

    for (i = 1; i < cnt_r + 1; i++)
        for (j = 0; j <= N - 1; j++)
            for (k = 0; k <= N - 1; k++) {
                s_local = s_local + A[i][j][k] * (i + 1) * (j + 1) * (k + 1) / (N3);
            }

    MPI_Reduce(&s_local, &s, 1 , MPI_DOUBLE, MPI_SUM, 0, global_comm);

    if (!rank) {
        printf("  S = %f\n", s);
    }
}

void pass_last_row() {
    if (rank)
        MPI_Irecv(A[0][0], N2, MPI_DOUBLE, rank - 1, TAG_PASS_LAST, global_comm, req_buf);
    if (rank != size - 1)
        MPI_Isend(A[cnt_r][0], N2, MPI_DOUBLE, rank + 1, TAG_PASS_LAST, global_comm, req_buf + 2);    
}

void pass_first_row() {
    if (rank != size - 1)
        MPI_Irecv(A[cnt_r + 1][0], N2, MPI_DOUBLE, rank + 1, TAG_PASS_FIRST, global_comm, req_buf + 3);    
    if (rank)
        MPI_Isend(A[1][0], N2, MPI_DOUBLE, rank - 1, TAG_PASS_FIRST, global_comm, req_buf + 1);
}

void wait_all() {
    int count = 4, shift = 0;
    if (!rank) {
        count -= 2;
        shift = 2;
    }
    if (rank == size - 1) {
        count -= 2;
    }
    
    MPI_Waitall(count, req_buf + shift, stat_buf);
}

void save_checkpoint() {
    MPI_File file;
    MPI_File_open(global_comm, file_path, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    for (i = 1; i < cnt_r + 1; i++) {
        MPI_File_write_at(file, sizeof(MPI_DOUBLE) * N2 * (fst_r - 1), A[i][0], N2, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(global_comm);
    MPI_File_close(&file);
}

void load_checkpoint() {
    MPI_File file;
    MPI_File_open(global_comm, file_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    for (i = 1; i < cnt_r + 1; i++) {
        MPI_File_read_at(file, sizeof(MPI_DOUBLE) * N2 * (fst_r - 1), A[i][0], N2, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(global_comm);
    MPI_File_close(&file);
}
