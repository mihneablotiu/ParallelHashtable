# Tema 3 - Parallel hashtable - ASC
# Blotiu Mihnea-Andrei - 333CA - 28.05.2023

The main idea of the homework was to implement a parallel hashtable that can insert, resize and get
elements using a GPU and the VRAM memory for the efective hashmap and using the CPU just to control
the execution of the operations.

That being said, my implementation is based on a hash function taken from [1] reference and uses
linear probing because in my opinion it is the simplest way to implement a hashmap where there is
only one value allowed per key. Unfortunatelly I have never heard before of Cuckoo or Bucketized
Cuckoo hashmap implementations and the time was very short so I just had to stick to the implementation
that I am used to.

* gpu_hashtable.hpp:

    - In the header file, I added a class called GpuHashTableInfo with the meaning of one pair
    inside of the hashmap. For this class I added getters and setters as the fields have been
    made private for encapsulation reasons.

    - Also, in the given GpuHashTable class I added some private fields needed for the implementation
    such as: the number of elements currently inside the hashmap, the maximum size of the hashmap,
    the load factor and the hashmap itself.

* gpu_hashtable.cu:

    - Inside the actual implementation file, I just completed the given functions and added 3 kernels
    and 2 additional functions that are going to be explained.

    - Firstly, in the constructor of the hashmap I had to think about how to put inside memory the
    hashmap. That being said, I decided that all the necesary information about the hashmap such
    as the size, load factor and the number of elements inside should be memorized in the normal
    RAM, but the actual hashmap, the keys and values themselves should be saved into the GPU
    VRAM Global Memory. That being said, I initialized the basic information inside RAM and allocated
    enough memory for the initial size hashmap in VRAM.

    - In the destructor, I just free the actual VRAM hashmap memory.

    - When inserting a vector of keys and values inside the hashmap, I firstly check what is going
    to be the maximum load factor if I am going to insert all of them inside the actual hashmap
    but without resizing. If that maximum load factor is going to be bigger than 90% (in my opinion
    a decent value), we are going to scale up the size of the hashmap with 1.5 in order not to 
    fall below 50% load factor. 
    
    - After resizing, we just copy the keys and values from host memory to device memory and we
    create one variable called newElements in the unified memory that is going to be both used
    by the device and host. This happens because even if we know that we have numKeys to insert
    we don't actually know how many will be new elements, because it is possible that some of the
    keys that we want to insert right now, have already been inserted before in the hashmap.

    - Then, we call the insert kernel and after that, we free the device memory and newElements'
    memory and return.

    - Inside the kernel, being a linear probing implementation, we just compute the hash value for
    each of the keys and limit it to the size of the hashmap. We try to atomically place the key
    to the position indicated by its hash (we do that because multiple keys might have the same
    hash). If that place was free or it contained the same key before, it means that we have to
    actually update the value as well. If not, we start going further one by one positions in
    the hashmap to the right until we find one free spot where we can put the key-value pair.

    - For the get operation, the procedure is simmilar to inserting with the difference that we
    have no atomic operations because we do not modify the same position of a vector in the global
    memory with two threads. That being said, we firstly copy the keys from host memory to device
    memory and we also do an unified memory alloc for the result_values because this vector is going
    to be returned back to the main function but it is going to be filled by a GPU kernel.

    - That being said, in the get kernel we just compute the hash of each of the keys and we go
    directly to the position indicated by the hash in the hashmap. If we find there the same key
    that we were searching for, we put the value in the result_values vector and then return.
    Otherwise, we do again a linear probing from the initial position until we find the key that
    we are searching for.

    - The last operation is the reshape one. Here, the logic behind is pretty simple. We just
    alloc a new hashmap with the dimension given as a parameter and we call the reshape kernel
    we traverse all of the keys and compute the hash for them again but now we limit the result
    to the new size of the hashmap. After that, we just do the same operations as in insert
    trying to fill the new hashmap with linear probing. After the kernel finished its execution
    we free the memory of the previous hashmap and then point the hashmap to the new VRAM
    memory location.

* Output performances:
    * !!! PS: I have also included in the archive a file called output_performance.txt that was done
    1 hour before loading because I saw that the actual checker is pretty undeterministic and I wanted
    to have a proof not knowing when the moodle checker is actually going to check my homework.

    - From that output we can learn that the algorithm is working pretty well because the load factor
    is never above 90% (the maximum value that I was thinking about).

    - Also we can see that all the tests match the minimum speed required for them, which will have
    all the tests to pass, but we can also see that the average speed for insert operations is much
    lower than the get operations.

    - The fact above in my opinion is normal and it happens because of two reasons. Firstly, the insert
    operations can have a reshape operation firstly which takes a lot of time. Then, I do not know how
    good is the hash function that I am using, but I am pretty sure that it makes a lot of collisions 
    when the hashmap becomes pretty full meaning that we actually have to traverse the whole hashmap
    to find a position for the new key.

* Conclusion:
    - To be honest, I am pretty proud of the implementation that I finally made because it was my
    first actually programming in CUDA and it ended up pretty well. I am pretty disappointed that
    always the end of the semester is very full because I would have liked to have a lot more time
    in order to search about the Cuckoo and Bucketized Cuckoo implementations in order to use them.
