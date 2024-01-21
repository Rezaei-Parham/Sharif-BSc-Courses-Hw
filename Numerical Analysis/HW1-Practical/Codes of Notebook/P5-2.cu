#include <stdio.h>
#include <stdlib.h>
// sizes for shared memory
#define trsize 4
#define matsize 1024

// kernel for transposing matrix
__global__ void transpose(float *a, float* c){
    // row and column for the current thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    c[col*matsize + row] = a[row*matsize + col];
}

//CUDA kernel for multiplication
__global__ void multiply(const float *a, const float *b, float *c, int m){
    // defining row and column based on current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < m){ // check for being inside the bounds
        float sum = 0;
        for (int i=0; i<m; i++){
            sum+= a[row*m+i] * b[i+m*col]; //dot product
        }
        c[row*m+col] = sum; //assign the value to the result
    }
}

void checkMultiplicationCorrect(float *p1, float *p2, float *p3, int m){
    for (int i=0; i<m; i++){
        for (int j=0; j<m; j++){
            float val = 0;
            for (int k=0; k<m; k++){
                val += p1[i*m+k] * p2[k*m+j];
            }
            // check for being close enough
            if (abs(p3[i*m+j] - val) > 0.0001){
                printf("\nwrong multiplication\n");
                return;
            }
        }
    }
    printf("\ncorrect multiplication\n");
}

int main(){
    int m = matsize;
    size_t size = m * m * sizeof(float);

    //allocate memory for host
    float *p1 = (float *)malloc(size);
    float *p2 = (float *)malloc(size);
    float *p3 = (float *)malloc(size);

    //initialize host memory
    for (int i=0; i<m*m; i++){
        p1[i] = (float)rand()/RAND_MAX;
        p2[i] = (float)rand()/RAND_MAX;
    }

    float *dp1, *dp2, *dp3, *dp4;
    cudaMalloc(&dp1, size);
    cudaMalloc(&dp2, size);
    cudaMalloc(&dp3, size);
    cudaMalloc(&dp4, size);

    // copy data to device from host
    cudaMemcpy(dp1, p1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dp2, p2, size, cudaMemcpyHostToDevice);

    //launch kernel
    int thr = trsize;
    int blc = m / thr;
    dim3 threads = dim3(thr, thr);
    dim3 blocks = dim3(blc, blc);
    // get the time of multiplication
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    cudaEventRecord(begin);
    transpose<<<blocks, threads>>>(dp2, dp4);
    multiply<<<blocks, threads>>>(dp1, dp4, dp3, m);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time = 0;
    cudaEventElapsedTime(&time, begin, end);
    // copy back
    cudaMemcpy(p3, dp3, size, cudaMemcpyDeviceToHost);

    printf("Matrix Multiplication size %d time %f\n",m, time/1000);
    checkMultiplicationCorrect(p1, p2, p3, m);
    //free memory
    cudaFree(dp1);
    cudaFree(dp2);
    cudaFree(dp3);
    free(p1);
    free(p2);
    free(p3);
}
