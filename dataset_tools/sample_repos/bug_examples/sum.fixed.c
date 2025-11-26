#include <stdio.h>

int sum(int *arr, int n) {
    int s = 0;
    for (int i = 0; i < n; ++i) { // fixed: use <
        s += arr[i];
    }
    return s;
}
