#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctime>
#include <sstream>
#include <string>
#include "test_map.hpp"
#include "gpu_hashtable.hpp"

using namespace std;

#define BLOCK_SIZE 256
#define MAX_LOAD_FACTOR 0.9
#define RESHAPE_FACTOR 1.5

// Source: https://burtleburtle.net/bob/hash/integer.html
__device__ uint32_t hashIntegers(uint32_t initialValue) {
    initialValue = (initialValue ^ 61) ^ (initialValue >> 16);
    initialValue = initialValue + (initialValue << 3);
    initialValue = initialValue ^ (initialValue >> 4);
    initialValue = initialValue * 0x27d4eb2d;
    initialValue = initialValue ^ (initialValue >> 15);
    return initialValue;
}

/**
 * The function that calculates the number of blocks needed for
 * running the kernel taking into consideration the number of elements
 * that has to be processed
*/
size_t calculateNumberOfBlocks(int elementsToProcess) {
	size_t blocks_no = elementsToProcess / BLOCK_SIZE;

	// In case the number of elements cannot be divided by the
	// block size we have to add one more block
	if (elementsToProcess % BLOCK_SIZE != 0) {
		blocks_no++;
	}

	return blocks_no;
}

/**
 * Function constructor GpuHashTable
 * Performs init operation for the fields of the GpuHashTable
 * class such as the initial size, the initial load factor,
 * the initial number of elements
 * 
 * It also allocates enough memory on VRAM for the initial
 * size of the hashmap
 */
GpuHashTable::GpuHashTable(int size) {
	this->currentNumberOfElements = 0;
	this->maximumSize = size;
	this->loadFactor = 0.0;
	this->elements = 0;
	cudaError_t returnValue;

	glbGpuAllocator->_cudaMalloc((void **) &(this->elements), size * sizeof(GpuHashTableInfo));
	DIE(this->elements == 0, "Error at allocating hashmap in VRAM");

	/* Here we have to initialize the memory to 0 as we will only have to store uint32_t values
	bigger than 0 so when a position has 0 in the hashmap we know that it is not occupied */
	returnValue = cudaMemset(this->elements, 0, size * sizeof(GpuHashTableInfo));
	DIE(returnValue != 0, "Error at cuda memset initial values in VRAM");
}

/**
 * Function desctructor GpuHashTable where we free the only allocated
 * memory inside the VRAM which is the hashmap itself
 */
GpuHashTable::~GpuHashTable() {
	cudaError_t returnValue = glbGpuAllocator->_cudaFree(this->elements);
	DIE(returnValue != 0, "Error at cuda Free hashmap in VRAM");
}

/**
 * The kernel function used for redimensioning the hashmap to a new size
*/
__global__ void hashmap_reshape(uint32_t oldSize, GpuHashTableInfo *newElements,
								GpuHashTableInfo *oldElements, uint32_t newSize) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < oldSize) {
		// If the initial element was 0, we don't have to move it to the new hashmap
		// because all the elements in the new hashmap are already 0
		if (oldElements[i].getKey() == 0) {
			return;
		}

		// Otherwise we just do the hash function for the old element and now we
		// limit it to the new dimension of the hashmap (which is a different
		// value so the final hash is going to be different)
		uint32_t hash = hashIntegers(oldElements[i].getKey());
		uint32_t currentHash = hash % newSize;

		// We check to see if the position where we want to put the key is empty
		// and if it is, we put it there and also put the corresponding value
		int oldKey = atomicCAS(newElements[currentHash].getKeyAddr(), 0, oldElements[i].getKey());

		if (oldKey == 0) {
			atomicExch(newElements[currentHash].getValueAddr(), oldElements[i].getValue());
		} else {
			int nextStep = 1;

			// Otherwise we start doing linear probing until we find an empty position
			// There we put the new key-value pair
			while (nextStep != newSize) {
				unsigned int currentPosition = (hash + nextStep) % newSize;

				oldKey = atomicCAS(newElements[currentPosition].getKeyAddr(), 0, oldElements[i].getKey());

				if (oldKey == 0) {
					atomicExch(newElements[currentPosition].getValueAddr(), oldElements[i].getValue());
					return;
				}

				nextStep++;
			}
		}
	}
}
/**
 * Function reshape
 * Performs resize of the hashtable based on load factor to a new size
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	// Firstly we completly allocate a new hashmap with the new size
	GpuHashTableInfo *newElements = 0;
	cudaError_t returnValue = glbGpuAllocator->_cudaMalloc((void **) &newElements,
														   numBucketsReshape * sizeof(GpuHashTableInfo));
	DIE(returnValue != 0, "Error at cudaMalloc for the new elements in reshape");

	// Then we initialize it to 0 everywhere because we can only have positive values
	// so 0 values means that the position has not been occupied yet
	returnValue = cudaMemset(newElements, 0, numBucketsReshape * sizeof(GpuHashTableInfo));
	DIE(returnValue != 0, "Error at cuda memset initial values in VRAM");

	size_t blocks_no = calculateNumberOfBlocks(this->maximumSize);
	// Then we call the kernel with blocks number being maximum size of the hashmap over
	// the size of one block because even though we can have less elements in the hashmap
	// than the maximum size we have no ideea at what position those are located so we 
	// actually have to traverse the whole hashmap
	hashmap_reshape<<<blocks_no, BLOCK_SIZE>>>(this->maximumSize, newElements,
										       this->elements, numBucketsReshape);
	returnValue = cudaDeviceSynchronize();
	DIE(returnValue != 0, "Error at cuda device synchronize after reshape kernel");

	// Here we free the old hashmap and update the fields with their new values
	returnValue = glbGpuAllocator->_cudaFree(this->elements);
	DIE(returnValue != 0, "Error at cuda Free for the old elements in VRAM");

	this->elements = newElements;
	this->maximumSize = numBucketsReshape;
	this->loadFactor = (double) this->currentNumberOfElements / this->maximumSize;
}

/**
 * The kernel function used to insert a vector of keys and values
 * into the hashmap using the GPU
*/
__global__ void hashmap_insert(const int *keys, const int *values,
							   const int maximumSize, GpuHashTableInfo *elements,
							   unsigned int *newElements, const int numKeys) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < numKeys) {
		// Firstly we calculate the current hash of the key and limit it to the
		// current maximum size of the hashmap and try to put it at the specific position
		uint32_t hash = hashIntegers(keys[i]);
		uint32_t currentHash = hash % maximumSize;

		int oldKey = atomicCAS(elements[currentHash].getKeyAddr(), 0, keys[i]);
		int oldValue;

		// If the position was previously empty or it contained exactly the same key
		// that we are trying to put right now, it means that we have to also update
		// the value of that specific key
		if (oldKey == 0 || oldKey == keys[i]) {
			oldValue = atomicExch(elements[currentHash].getValueAddr(), values[i]);

			// If the old value was 0 it means that it is a completly new key-value
			// pair added to the hashmap so we also have to increment the number of
			// new values added to the hashmap
			if (oldValue == 0) {
				atomicInc(newElements, maximumSize);
			}

			return;
		} else {
			int nextStep = 1;

			// Otherwise we start doing linear probing until we find an empty position
			// There we put the new key-value pair
			while (nextStep != maximumSize) {
				unsigned int currentPosition = (hash + nextStep) % maximumSize;

				oldKey = atomicCAS(elements[currentPosition].getKeyAddr(), 0, keys[i]);

				if (oldKey == 0 || oldKey == keys[i]) {
					oldValue = atomicExch(elements[currentPosition].getValueAddr(), values[i]);

					// If the old value was 0 it means that it is a completly new key-value
					// pair added to the hashmap so we also have to increment the number of
					// new values added to the hashmap
					if (oldValue == 0) {
						atomicInc(newElements, maximumSize);
					}

					return;
				}

				nextStep++;
			}
		}
	}

	return;
}

/**
 * Function insertBatch
 * Inserts a batch of key:value, using GPU and wrapper allocators
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	// Here, firstly we verify what is going to be the maximum load factor if we are going
	// to add all the given keys to the hashmap. I am saying here the `maximum` load factor
	// because some of the keys might be identical to some of the keys that are already
	// inside the hashmap so in that case, we just have to update the value so no more keys
	// added to the hashmap. If the maximum possible load factor is going to exceed for example
	// 90%, we will have to resize it to 1.5 * currentSize in order to have a balanced hashmap
	// and not to fall with the load factor under 50%
	double possibleMaxLoadFactor = (double) (this->currentNumberOfElements + numKeys) / this->maximumSize;
	if (possibleMaxLoadFactor >= MAX_LOAD_FACTOR) {
		this->reshape(this->maximumSize * RESHAPE_FACTOR);
	}

	int *device_keys = 0;
	int *device_values = 0;

	// Here we copy the keys and the values from host to device
	glbGpuAllocator->_cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	glbGpuAllocator->_cudaMalloc((void **) &device_values, numKeys * sizeof(int));
	DIE(device_keys == 0 || device_values == 0, "Error at cudaMalloc for keys/values in insert");

	cudaError_t returnValue = cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(returnValue != 0, "Error at cuda memcpy host keys to device keys in VRAM");

	returnValue = cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(returnValue != 0, "Error at cuda memcpy host values to device values in VRAM");

	// We calculate the blocks number as being the number of keys over the size of one block
	// that is because we have to add maximum numKeys keys inside the hashmap
	size_t blocks_no = calculateNumberOfBlocks(numKeys);

	// This is going to be a cudaMallocManaged integer value that is going to be allocated in the
	// unified memory. This is happening because it is going to be changed by the insert kernel
	// each time when a new key-pair value is added in the hashmap and it is also going to be
	// verified by the host in order to know how many actual key were added durring the kernel exec
	unsigned int *newElements = 0;
	glbGpuAllocator->_cudaMallocManaged((void **) &newElements, sizeof(unsigned int));
	DIE(newElements == 0, "Error at cudaMalloc for the number of new inserted elements");
	*newElements = 0;

	hashmap_insert<<<blocks_no, BLOCK_SIZE>>>(device_keys, device_values, this->maximumSize,
	 										  this->elements, newElements, numKeys);
	returnValue = cudaDeviceSynchronize();
	DIE(returnValue != 0, "Error at cuda device synchronize after insert kernel");

	// Here, if there were any new key-value pairs added to the hashmap we increment the
	// current number of elements and recalculate the load factor
	if (*newElements != 0) {
		this->currentNumberOfElements += *newElements;
		this->loadFactor = (double) this->currentNumberOfElements / this->maximumSize;
	}

	// We free the device keys, values and the new elements int value and exit the function
	returnValue = glbGpuAllocator->_cudaFree(device_keys);
	DIE(returnValue != 0, "Error at cuda Free device keys in VRAM");

	returnValue = glbGpuAllocator->_cudaFree(device_values);
	DIE(returnValue != 0, "Error at cuda Free device values in VRAM");

	returnValue = glbGpuAllocator->_cudaFree(newElements);
	DIE(returnValue != 0, "Error at cuda Free unified memory for the new inserted elements");

	return true;
}

/**
 * The kernel function used to get a vector of values starting from a vector
 * of keys from the hashmap
*/
__global__ void hashmap_get(const int *keys, int *result_values,
					        const int maximumSize, GpuHashTableInfo *elements,
					        const int numKeys)
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	// Here we do the same operation as in inserting. We firstly just calculate
	// the hash of the current key that we are trying to find. And then we try
	// to see if the key from the position of the hash is the one we are actually
	// searching for. If it is, we just take the value from the same position and return.
	if (i < numKeys) {
		uint32_t hash = hashIntegers(keys[i]);
		uint32_t currentHash = hash % maximumSize;

		int oldKey = elements[currentHash].getKey();
		int oldValue;

		if (oldKey == keys[i]) {
			oldValue = elements[currentHash].getValue();
			result_values[i] = oldValue;
			return;

		} else {
			int nextStep = 1;

			// Otherwise we start doing linear probing until we find a position
			// where the key from that position is the same with the key we are
			// searching for
			while (nextStep != maximumSize) {
				unsigned int currentPosition = (hash + nextStep) % maximumSize;

				oldKey = elements[currentPosition].getKey();

				if (oldKey == keys[i]) {
					oldValue = elements[currentPosition].getValue();
					result_values[i] = oldValue;
					return;
				}

				nextStep++;
			}
		}
	}
}

/**
 * Function getBatch
 * Gets a batch of key:value, using GPU
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *device_keys = 0;
	int *result_values = 0;

	// Firstly we allocate the device_keys vector and copy the keys from the host to device.
	cudaError_t returnValue = glbGpuAllocator->_cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	DIE(returnValue != 0, "Error at cuda malloc for the device keys in get batch");

	returnValue = cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(returnValue != 0, "Error at cuda memcpy for the keys into device keys in get batch");

	// Then, here we allocate a result_values vector with cudaMallocManaged because it is going to
	// be populated by the kernel but it is going to be returned from this host function and also
	// used further in main.
	returnValue = glbGpuAllocator->_cudaMallocManaged((void **) &result_values, numKeys * sizeof(int));
	DIE(returnValue != 0, "Error at cuda malloc for the result values in get batch");

	// We calculate the blocks number as being the number of keys over the size of one block
	// that is because we have to get exactly numKeys keys from the hashmap
	size_t blocks_no = calculateNumberOfBlocks(numKeys);

	// We execute the kernel, free the device keys and return the newly populated result_values
	// from the kernel.
	hashmap_get<<<blocks_no, BLOCK_SIZE>>>(device_keys, result_values, this->maximumSize,
										   this->elements, numKeys);
	returnValue = cudaDeviceSynchronize();
	DIE(returnValue != 0, "Error at cuda device synchronize after insert kernel");

	returnValue = glbGpuAllocator->_cudaFree(device_keys);
	DIE(returnValue != 0, "Error at cuda Free device values in VRAM");

	return result_values;
}
