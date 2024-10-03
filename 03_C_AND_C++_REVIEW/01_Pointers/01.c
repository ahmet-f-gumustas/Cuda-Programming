#include "stdio.h" // Standart input/output header file (for printf)

// & "adress of" operator
// * "dereference" operator

int main() {

    int x = 10;
    int* ptr = &x;  // & is used to get the memory address of a variable
    printf("Address of x: %p\n", ptr);
    printf("Value of x: %d\n", ptr);    // this is no dereference = somting : -18124135 bla bla
    printf("Value of with Adress of: %d\n", *ptr);   // this is a de reference = 10
    // * in the prev lien is used to get the value of
    // the memory address stored in ptr (dereferencing)

    printf("\n");

    char y = "ahmet faruk";
    char* ptr_str = &y;
    printf("address of y: %p\n", ptr_str);
    printf("char of x: %d\n", ptr_str);
    printf("Char of with address of: %d\n", *ptr_str);

    printf("\n");

    float z = 0.123;
    float* ptr_float = &z;
    printf("address of y: %p\n", ptr_float);
    printf("char of x: %d\n", ptr_float);
    printf("Char of with address of: %d\n", *ptr_float);

    printf("\n");

}
