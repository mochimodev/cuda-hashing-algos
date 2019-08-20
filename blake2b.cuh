/*
 * blake2b.cuh CUDA Implementation of BLAKE2B Hashing
 *
 * Date: 12 June 2019
 * Revision: 1
 *
 * This file is released into the Public Domain.
 */


#pragma once
#include "config.h"
void mcm_cuda_blake2b_hash_batch(BYTE* key, WORD keylen, BYTE * in, WORD inlen, BYTE * out, WORD n_outbit, WORD n_batch);
