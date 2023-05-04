#pragma once
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

using namespace std;
// #define ENABLE_TIME_INFO 0

#define TIME_LIMIT 300 // enumeration time limit, unit: s

typedef unsigned int Label;
typedef uint16_t Value;

typedef unsigned int Vertex;
#define MAX_VERTEX_ID 0xffffffff // mush be consistent with the type Vertex

#define OVERFLOW_CHECK 0

#define FAILING_SET_PRUNING 0

// #define CORE_DECOMPOSITION 0

#define MAX_QUERY_SIZE 60

// 0 stands for no optimization; 1 for AVX2; 2 for AVX512
#define AVX 2

#define MAX_SAMPLE_NUM 20000

// static pthread_barrier_t barrier;
// static string mask_file_name = "";