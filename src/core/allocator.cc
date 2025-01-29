#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        // Naive: Iterate through the free blocks to find a suitable block
        for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it) {
            size_t block_start = it->first;
            size_t block_size = it->second;
            if (block_size >= size) {
                // Allocate memory from this block
                size_t allocated_addr = block_start;
                // Update the free block map
                if (block_size > size) {
                    // Reduce the size of the free block
                    free_blocks[block_start + size] = block_size - size;
                }
                free_blocks.erase(it);
                // Update used memory
                used += size;
                if (used > peak) {
                    peak = used;
                }
                return allocated_addr;
            }
        }
        //Red-black tree
    // If no suitable block is found, return 0 (or handle out-of-memory case)
        std::cout << "No suitable block found" << std::endl;
        return 0;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);
        
        // Update used memory
        used -= size;
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        // Add the freed block to the free_blocks map
        auto it = free_blocks.lower_bound(addr); // Find the first block with address >= addr

        // Try to merge with the previous block
        if (it != free_blocks.begin()) {
            auto prev = std::prev(it); // Get the previous block
            if (prev->first + prev->second == addr) {
                addr = prev->first;
                size += prev->second;
                free_blocks.erase(prev);
            }
        }

        // Try to merge with the next block
        if (it != free_blocks.end() && addr + size == it->first) {
            size += it->second;
            free_blocks.erase(it);
        }

        // Insert the merged block into the free_blocks map
        free_blocks[addr] = size;

    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
