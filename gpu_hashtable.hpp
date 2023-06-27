#ifndef _HASHCPU_
#define _HASHCPU_

#include <inttypes.h>

/**
 * The class used for one element of the hashmap
 * a pair of a key-value
*/
class GpuHashTableInfo
{
	private:
		uint32_t key;
		uint32_t value;

	public:
		__device__ uint32_t *getKeyAddr() {
			return &this->key;
		}

		__device__ uint32_t *getValueAddr() {
			return &this->value;
		}

		__device__ uint32_t getKey() {
			return this->key;
		}

		__device__ uint32_t getValue() {
			return this->value;
		}

		__device__ void setKey(uint32_t key) {
			this->key = key;
			return;
		}

		__device__ void setValue(uint32_t value) {
			this->value = value;
			return;
		}
};

/**
 * Class GpuHashTable to implement functions
 */
class GpuHashTable
{
	private:
		int maximumSize;
		int currentNumberOfElements;
		double loadFactor;
		GpuHashTableInfo *elements;

	public:
		GpuHashTable(int size);
		void reshape(int sizeReshape);

		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);

		~GpuHashTable();
};

#endif
