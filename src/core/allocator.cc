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
    Allocator::~Allocator() {
        if (this->ptr != nullptr) {
            runtime->dealloc(this->ptr);
        }
    }
    // Allocates a block of memory of the given size
    size_t Allocator::alloc(size_t size) {
        // Ensure that the pointer is null before allocation
        IT_ASSERT(this->ptr == nullptr);

        // Pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // Update the used memory counter
        this->used += size;

        // Iterate through the list of free blocks to find a suitable block
        for (auto it = this->free_blocks.begin(); it != this->free_blocks.end(); it++) {
            // Check if the current block is large enough
            if (it->second >= size) {
                size_t addr = it->first; // Address of the free block
                size_t space = it->second - size; // Remaining space after allocation

                // Remove the current block from the free list
                this->free_blocks.erase(it);

                // If there is remaining space, add it back to the free list
                if (space > 0) {
                    this->free_blocks[addr + size] = space;
                }

                // Return the address of the allocated block
                return addr;
            }
        }

        // If no suitable block was found, increase the peak memory usage
        this->peak += size;

        // Return the address of the newly allocated block
        return this->peak - size;
    }


    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // Update used memory
        used -= size;

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
