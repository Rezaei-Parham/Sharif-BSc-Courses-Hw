#include <stdio.h>
#include <stdlib.h>

#define matsize 1024
#define tsize 32
//CUDA kernel
__global__ void multiply(const float *a, const float *b, float *c, int m){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    __shared__ float s_a[tsize][tsize];
    __shared__ float s_b[tsize][tsize];

    for(int i=0;i < m/tsize; i++){
        s_a[threadIdx.y][threadIdx.x] = a[row*m + threadIdx.x + i*tsize];
        s_b[threadIdx.y][threadIdx.x] = b[m*(threadIdx.y+i*tsize)+col];
        __syncthreads();
        for(int p=0;p<tsize;p++){
            sum += s_a[threadIdx.y][p]*s_b[p][threadIdx.x];
        }
        __syncthreads();
    }
    if (col < m && row < m) c[m*row + col] = sum;
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
    float *dev1, *dev2, *dev3;
    cudaMalloc(&dev1, size);
    cudaMalloc(&dev2, size);
    cudaMalloc(&dev3, size);

    // copy data to device from host
    cudaMemcpy(dev1, p1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev2, p2, size, cudaMemcpyHostToDevice);

    //launch kernel
    int thr = tsize;
    int blc = m / thr;
    dim3 threads = dim3(thr, thr);
    dim3 blocks = dim3(blc, blc);
    // get the time of multiplication
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    multiply<<<blocks, threads>>>(dev1, dev2, dev3, m);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    // copy back
    cudaMemcpy(p3, dev3, size, cudaMemcpyDeviceToHost);

    printf("time of multiplication with gpu: %f\n", time/1000);
    // checkMultiplicationCorrect(p1, p2, p3, m);
    //free memory
    cudaFree(dev1);
    cudaFree(dev2);
    cudaFree(dev3);
    free(p1);
    free(p2);
    free(p3);
}
