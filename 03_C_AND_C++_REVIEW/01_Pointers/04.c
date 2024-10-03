// Purpuse: Demonstrate NULL pointer initialization and safe usage

// Key points:
// 1. Initialize pointers to NULL when they don't yet point to valid data.
// 2. Check pointers for NULL before using to avoid crashes
// 3. NULL checks allow graceful handling of uninitialized or failed allocation

#include <stdio.h>
#include <stdlib.h>

int main() {
    // Initizlize pointer to NULL
    int* ptr = NULL;
    printf("1. Initial ptr value: %p\n", (void*)ptr);

    // Check for NULL before using
    if (ptr = NULL){
        printf("2. ptr is NULL, cannot dereference\n");
    }

    // Allocate memory
    ptr = (int*) malloc(sizeof(int));
    if (ptr == NULL){
        printf("3. Memory allocation failed\n");
        return 1;
    }

    return 0;
}