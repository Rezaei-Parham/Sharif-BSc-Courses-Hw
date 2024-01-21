#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void multiply(float **p1, float **p2, float **p3, int m);
int main(){
    int m = 1024;
    int fpsize = m * sizeof(float *);
    int fsize = m * sizeof(float);
    // first matrix
    float **p1 = (float **)malloc(fpsize);
    // second matrix
    float **p2 = (float **)malloc(fpsize);
    // result matrix
    float **p3 = (float **)malloc(fpsize);
    // allocating each row of the matrix
    for (int i = 0; i < m; i++){
        p1[i] = (float *)malloc(fsize);
        p2[i] = (float *)malloc(fsize);
        p3[i] = (float *)calloc(m,sizeof(float)); // calloc so no need to initialize to 0
    }
    // fill matrices randomly
    for(int i = 0; i < m; i++){
        for(int j = 0; j < m; j++){
            p1[i][j] = (float)rand() / RAND_MAX;
            p2[i][j] = (float)rand() / RAND_MAX;
        }
    }
    // calculating the time
    clock_t start,end;
    start = clock();
    // doing the multiplication
    multiply(p1, p2, p3, m);
    end = clock();
    // calculate the time it took
    double time = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("CPU Matrix Multiplication Size %d took %f seconds\n", m, time);

    // free the allocated memory
    for(int i = 0; i < m; i++){
        free(p1[i]);
        free(p2[i]);
        free(p3[i]);
    }
    free(p1);
    free(p2);
    free(p3);
    return 0;
}


void multiply(float **p1, float **p2, float **p3, int m){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < m; j++){
            for(int l = 0; l < m; l++){
                p3[i][j] += p1[i][l] * p2[l][j];
            }
        }
    }
}
