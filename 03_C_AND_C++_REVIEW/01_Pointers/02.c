#include <stdio.h>

int main (){

    int value =42;
    int* ptr = &value;
    int** ptr2 = &ptr;
    int*** ptr3 = &ptr2;

    printf("Value : %d\n ", ***ptr3);

    return 0;
}