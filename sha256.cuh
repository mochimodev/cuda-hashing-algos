/*
 * sha256.cuh CUDA Implementation of SHA256 Hashing    
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


#pragma once
#include "config.h"
void mcm_cuda_sha256_hash_batch(BYTE* in, WORD inlen, BYTE* out, WORD n_batch);
