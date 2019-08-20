/*
 * md2.cu CUDA Implementation of MD2 digest
 *
 * Date: 12 June 2019
 * Revision: 1
 * 
 * Based on the public domain Reference Implementation in C, by
 * Brad Conte, original code here:
 *
 * https://github.com/B-Con/crypto-algorithms
 *
 * This file is released into the Public Domain.
 */


/*************************** HEADER FILES ***************************/
#include <stdlib.h>
#include <memory.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
extern "C" {
#include "md2.cuh"
}
#define MD2_BLOCK_SIZE 16
/**************************** STRUCT ********************************/
typedef struct {
	BYTE data[16];
	BYTE state[48];
	BYTE checksum[16];
	int len;
} CUDA_MD2_CTX;

/**************************** VARIABLES *****************************/
__constant__ BYTE s[256] = {
	41, 46, 67, 201, 162, 216, 124, 1, 61, 54, 84, 161, 236, 240, 6,
	19, 98, 167, 5, 243, 192, 199, 115, 140, 152, 147, 43, 217, 188,
	76, 130, 202, 30, 155, 87, 60, 253, 212, 224, 22, 103, 66, 111, 24,
	138, 23, 229, 18, 190, 78, 196, 214, 218, 158, 222, 73, 160, 251,
	245, 142, 187, 47, 238, 122, 169, 104, 121, 145, 21, 178, 7, 63,
	148, 194, 16, 137, 11, 34, 95, 33, 128, 127, 93, 154, 90, 144, 50,
	39, 53, 62, 204, 231, 191, 247, 151, 3, 255, 25, 48, 179, 72, 165,
	181, 209, 215, 94, 146, 42, 172, 86, 170, 198, 79, 184, 56, 210,
	150, 164, 125, 182, 118, 252, 107, 226, 156, 116, 4, 241, 69, 157,
	112, 89, 100, 113, 135, 32, 134, 91, 207, 101, 230, 45, 168, 2, 27,
	96, 37, 173, 174, 176, 185, 246, 28, 70, 97, 105, 52, 64, 126, 15,
	85, 71, 163, 35, 221, 81, 175, 58, 195, 92, 249, 206, 186, 197,
	234, 38, 44, 83, 13, 110, 133, 40, 132, 9, 211, 223, 205, 244, 65,
	129, 77, 82, 106, 220, 55, 200, 108, 193, 171, 250, 36, 225, 123,
	8, 12, 189, 177, 74, 120, 136, 149, 139, 227, 99, 232, 109, 233,
	203, 213, 254, 59, 0, 29, 57, 242, 239, 183, 14, 102, 88, 208, 228,
	166, 119, 114, 248, 235, 117, 75, 10, 49, 68, 80, 180, 143, 237,
	31, 26, 219, 153, 141, 51, 159, 17, 131, 20
};

/*********************** FUNCTION DEFINITIONS ***********************/
__device__ void cuda_md2_transform(CUDA_MD2_CTX *ctx, BYTE data[])
{
	int j,k,t;

	//memcpy(&ctx->state[16], data);
	for (j=0; j < 16; ++j) {
		ctx->state[j + 16] = data[j];
		ctx->state[j + 32] = (ctx->state[j+16] ^ ctx->state[j]);
	}

	t = 0;
	for (j = 0; j < 18; ++j) {
		for (k = 0; k < 48; ++k) {
			ctx->state[k] ^= s[t];
			t = ctx->state[k];
		}
		t = (t+j) & 0xFF;
	}

	t = ctx->checksum[15];
	for (j=0; j < 16; ++j) {
		ctx->checksum[j] ^= s[data[j] ^ t];
		t = ctx->checksum[j];
	}
}

__device__ void cuda_md2_init(CUDA_MD2_CTX *ctx)
{
	int i;

	for (i=0; i < 48; ++i)
		ctx->state[i] = 0;
	for (i=0; i < 16; ++i)
		ctx->checksum[i] = 0;
	ctx->len = 0;
}

__device__ void cuda_md2_update(CUDA_MD2_CTX *ctx, const BYTE data[], size_t len)
{
	size_t i;

	for (i = 0; i < len; ++i) {
		ctx->data[ctx->len] = data[i];
		ctx->len++;
		if (ctx->len == MD2_BLOCK_SIZE) {
			cuda_md2_transform(ctx, ctx->data);
			ctx->len = 0;
		}
	}
}

__device__ void cuda_md2_final(CUDA_MD2_CTX *ctx, BYTE hash[])
{
	int to_pad;

	to_pad = MD2_BLOCK_SIZE - ctx->len;

	while (ctx->len < MD2_BLOCK_SIZE)
		ctx->data[ctx->len++] = to_pad;

	cuda_md2_transform(ctx, ctx->data);
	cuda_md2_transform(ctx, ctx->checksum);

	memcpy(hash, ctx->state, MD2_BLOCK_SIZE);
}

__global__ void kernel_md2_hash(BYTE* indata, WORD inlen, BYTE* outdata, WORD n_batch)
{
	WORD thread = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread >= n_batch)
	{
		return;
	}
	BYTE* in = indata  + thread * inlen;
	BYTE* out = outdata  + thread * MD2_BLOCK_SIZE;
	CUDA_MD2_CTX ctx;
	cuda_md2_init(&ctx);
	cuda_md2_update(&ctx, in, inlen);
	cuda_md2_final(&ctx, out);
}
extern "C" {
void mcm_cuda_md2_hash_batch(BYTE *in, WORD inlen, BYTE *out, WORD n_batch) {
	BYTE *cuda_indata;
	BYTE *cuda_outdata;
	cudaMalloc(&cuda_indata, inlen * n_batch);
	cudaMalloc(&cuda_outdata, MD2_BLOCK_SIZE * n_batch);
	cudaMemcpy(cuda_indata, in, inlen * n_batch, cudaMemcpyHostToDevice);

	WORD thread = 256;
	WORD block = (n_batch + thread - 1) / thread;

	kernel_md2_hash << < block, thread >> > (cuda_indata, inlen, cuda_outdata, n_batch);
	cudaMemcpy(out, cuda_outdata, MD2_BLOCK_SIZE * n_batch, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error cuda md2 hash: %s \n", cudaGetErrorString(error));
	}
	cudaFree(cuda_indata);
	cudaFree(cuda_outdata);
}
}
