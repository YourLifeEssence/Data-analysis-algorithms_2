#include <stdio.h>
#include <omp.h>
#include <math.h>

#define N 5

//Разложение на сумму квадратов
void sum_of_squares(int n) {
    int found = 0;
    for(int i = 0; i*i <= n && !found; i++) {
        for(int j = 0; j*j <= n && !found; j++) {
            for(int k = 0; k*k <= n && !found; k++) {
                int l2 = n - i*i - j*j - k*k;
                int l = (int)sqrt(l2);
                if(l*l == l2) {
                    printf("Sum of squares of %d = %d^2 + %d^2 + %d^2 + %d^2\n", n, i,j,k,l);
                    found = 1;
                }
            }
        }
    }
}

//n-е число Фибоначчи
long long fibonacci(int n) {
    if(n <= 1) return n;
    long long a = 0, b = 1, c;
    for(int i = 2; i <= n; i++) {
        c = a + b;
        a = b;
        b = c;
    }
    return b;
}

//n-й простой
int nth_prime(int n) {
    if(n == 1) return 2;
    int count = 1;
    int num = 3;
    while(count < n) {
        int prime = 1;
        for(int i = 2; i*i <= num; i++) {
            if(num % i == 0) { prime = 0; break; }
        }
        if(prime) count++;
        if(count < n) num += 2;
    }
    return num;
}

//Сумма делителей
int sum_of_divisors(int n) {
    int sum = 0;
    for(int i = 1; i <= n; i++) {
        if(n % i == 0) sum += i;
    }
    return sum;
}

int main() {
    int arr[N] = {10, 15, 20, 25, 30};

    for(int i = 0; i < N; ++i) {
        int n = arr[i];

        #pragma omp parallel sections
        {
            #pragma omp section
            sum_of_squares(n);

            #pragma omp section
            printf("Fibonacci(%d) = %lld\n", n, fibonacci(n));

            #pragma omp section
            printf("%d-th prime = %d\n", n, nth_prime(n));

            #pragma omp section
            printf("Sum of divisors of %d = %d\n", n, sum_of_divisors(n));
        }

        printf("\n");
    }
    return 0;
}