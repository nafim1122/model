import { NextRequest } from 'next/server';

const MODEL_ENDPOINT = process.env.MODEL_ENDPOINT || 'http://localhost:8000/generate';
const USE_MOCK = process.env.USE_MOCK !== 'false';

const C_EXAMPLES: Record<string, string> = {
  prime: `#include <stdio.h>
#include <stdbool.h>

bool isPrime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

int main() {
    int num;
    printf("Enter a number: ");
    scanf("%d", &num);
    printf("%d %s a prime number.\\n", num, isPrime(num) ? "is" : "is not");
    return 0;
}`,
  reverse: `#include <stdio.h>
#include <string.h>

void reverseString(char *str) {
    int left = 0, right = strlen(str) - 1;
    while (left < right) {
        char temp = str[left];
        str[left++] = str[right];
        str[right--] = temp;
    }
}

int main() {
    char str[100];
    printf("Enter a string: ");
    fgets(str, sizeof(str), stdin);
    str[strcspn(str, "\\n")] = 0;
    reverseString(str);
    printf("Reversed: %s\\n", str);
    return 0;
}`,
  fibonacci: `#include <stdio.h>

void printFibonacci(int n) {
    long long a = 0, b = 1;
    for (int i = 0; i < n; i++) {
        printf("%lld ", a);
        long long next = a + b;
        a = b;
        b = next;
    }
    printf("\\n");
}

int main() {
    int n;
    printf("Enter number of terms: ");
    scanf("%d", &n);
    printFibonacci(n);
    return 0;
}`,
  sort: `#include <stdio.h>

void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}

int main() {
    int arr[] = {64, 34, 25, 12, 22, 11, 90};
    int n = sizeof(arr)/sizeof(arr[0]);
    bubbleSort(arr, n);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    return 0;
}`,
  largest: `#include <stdio.h>

int findLargest(int arr[], int n) {
    int max = arr[0];
    for (int i = 1; i < n; i++)
        if (arr[i] > max) max = arr[i];
    return max;
}

int main() {
    int arr[] = {10, 324, 45, 90, 9808};
    int n = sizeof(arr)/sizeof(arr[0]);
    printf("Largest: %d\\n", findLargest(arr, n));
    return 0;
}`,
  vowels: `#include <stdio.h>
#include <ctype.h>

int countVowels(const char *str) {
    int count = 0;
    while (*str) {
        char c = tolower(*str++);
        if (c=='a'||c=='e'||c=='i'||c=='o'||c=='u') count++;
    }
    return count;
}

int main() {
    char str[256];
    printf("Enter string: ");
    fgets(str, sizeof(str), stdin);
    printf("Vowels: %d\\n", countVowels(str));
    return 0;
}`,
  factorial: `#include <stdio.h>

long long factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int main() {
    int n;
    printf("Enter number: ");
    scanf("%d", &n);
    printf("%d! = %lld\\n", n, factorial(n));
    return 0;
}`,
  palindrome: `#include <stdio.h>

int isPalindrome(int n) {
    int orig = n, rev = 0;
    while (n > 0) {
        rev = rev * 10 + n % 10;
        n /= 10;
    }
    return orig == rev;
}

int main() {
    int num;
    printf("Enter number: ");
    scanf("%d", &num);
    printf("%d %s a palindrome\\n", num, isPalindrome(num) ? "is" : "is not");
    return 0;
}`,
  gcd: `#include <stdio.h>

int gcd(int a, int b) {
    while (b) { int t = b; b = a % b; a = t; }
    return a;
}

int main() {
    int a, b;
    printf("Enter two numbers: ");
    scanf("%d %d", &a, &b);
    printf("GCD: %d\\n", gcd(a, b));
    return 0;
}`,
  swap: `#include <stdio.h>

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int main() {
    int x = 10, y = 20;
    printf("Before: x=%d, y=%d\\n", x, y);
    swap(&x, &y);
    printf("After: x=%d, y=%d\\n", x, y);
    return 0;
}`,
  binary: `#include <stdio.h>

int binarySearch(int arr[], int n, int key) {
    int left = 0, right = n - 1;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (arr[mid] == key) return mid;
        if (arr[mid] < key) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

int main() {
    int arr[] = {2, 5, 8, 12, 16, 23, 38};
    int n = sizeof(arr)/sizeof(arr[0]);
    printf("Index of 12: %d\\n", binarySearch(arr, n, 12));
    return 0;
}`,
  pointer: `#include <stdio.h>

int main() {
    int var = 42;
    int *ptr = &var;
    
    printf("Value: %d\\n", var);
    printf("Address: %p\\n", (void*)&var);
    printf("Pointer value: %p\\n", (void*)ptr);
    printf("Dereferenced: %d\\n", *ptr);
    
    *ptr = 100;
    printf("After *ptr=100: var=%d\\n", var);
    return 0;
}`,
  struct: `#include <stdio.h>
#include <string.h>

struct Student {
    int id;
    char name[50];
    float gpa;
};

int main() {
    struct Student s = {101, "John Doe", 3.85};
    printf("ID: %d\\nName: %s\\nGPA: %.2f\\n", s.id, s.name, s.gpa);
    return 0;
}`,
  calculator: `#include <stdio.h>

int main() {
    double a, b;
    char op;
    printf("Enter expression (e.g. 5+3): ");
    scanf("%lf %c %lf", &a, &op, &b);
    
    switch(op) {
        case '+': printf("%.2f\\n", a+b); break;
        case '-': printf("%.2f\\n", a-b); break;
        case '*': printf("%.2f\\n", a*b); break;
        case '/': printf("%.2f\\n", b?a/b:0); break;
        default: printf("Unknown operator\\n");
    }
    return 0;
}`,
  armstrong: `#include <stdio.h>
#include <math.h>

int isArmstrong(int n) {
    int orig = n, sum = 0, digits = 0;
    for (int t = n; t; t /= 10) digits++;
    for (int t = n; t; t /= 10) sum += pow(t % 10, digits);
    return sum == orig;
}

int main() {
    int n;
    printf("Enter number: ");
    scanf("%d", &n);
    printf("%d %s Armstrong\\n", n, isArmstrong(n) ? "is" : "is not");
    return 0;
}`,
  matrix: `#include <stdio.h>

int main() {
    int a[2][2] = {{1,2},{3,4}};
    int b[2][2] = {{5,6},{7,8}};
    int c[2][2] = {0};
    
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                c[i][j] += a[i][k] * b[k][j];
    
    printf("Result:\\n");
    for (int i = 0; i < 2; i++)
        printf("%d %d\\n", c[i][0], c[i][1]);
    return 0;
}`,
  malloc: `#include <stdio.h>
#include <stdlib.h>

int main() {
    int n = 5;
    int *arr = (int*)malloc(n * sizeof(int));
    
    for (int i = 0; i < n; i++) arr[i] = i * 10;
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    
    free(arr);
    return 0;
}`,
  file: `#include <stdio.h>

int main() {
    FILE *f = fopen("test.txt", "w");
    if (f) {
        fprintf(f, "Hello, File!\\n");
        fclose(f);
        printf("Written successfully\\n");
    }
    return 0;
}`,
  linked: `#include <stdio.h>
#include <stdlib.h>

struct Node {
    int data;
    struct Node *next;
};

int main() {
    struct Node *head = malloc(sizeof(struct Node));
    head->data = 1;
    head->next = malloc(sizeof(struct Node));
    head->next->data = 2;
    head->next->next = NULL;
    
    for (struct Node *p = head; p; p = p->next)
        printf("%d -> ", p->data);
    printf("NULL\\n");
    return 0;
}`,
};

function matchCode(prompt: string): string {
  const p = prompt.toLowerCase();
  const matches: [string[], string][] = [
    [['prime'], 'prime'], [['reverse', 'string'], 'reverse'],
    [['fibonacci', 'fibo'], 'fibonacci'], [['sort', 'bubble'], 'sort'],
    [['largest', 'maximum', 'array'], 'largest'], [['vowel'], 'vowels'],
    [['factorial', 'recursion'], 'factorial'], [['palindrome'], 'palindrome'],
    [['gcd', 'greatest common'], 'gcd'], [['swap'], 'swap'],
    [['binary search'], 'binary'], [['pointer'], 'pointer'],
    [['struct'], 'struct'], [['calculator', 'switch'], 'calculator'],
    [['armstrong'], 'armstrong'], [['matrix'], 'matrix'],
    [['malloc', 'dynamic'], 'malloc'], [['file'], 'file'],
    [['linked list'], 'linked'],
  ];
  for (const [keys, code] of matches)
    if (keys.some(k => p.includes(k)) && C_EXAMPLES[code]) return C_EXAMPLES[code];
  return `#include <stdio.h>\n\nint main() {\n    // ${prompt.slice(0,40)}\n    printf("Hello!\\n");\n    return 0;\n}`;
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { prompt } = body;

    if (!prompt) {
      return new Response(JSON.stringify({ error: 'Prompt is required' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    const code = matchCode(prompt);
    const response = '```c\n' + code + '\n```';

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        for (let i = 0; i < response.length; i++) {
          controller.enqueue(encoder.encode(response[i]));
          await new Promise(resolve => setTimeout(resolve, 3));
        }
        controller.close();
      },
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Transfer-Encoding': 'chunked',
      },
    });
  } catch (error) {
    console.error('API Error:', error);
    return new Response(JSON.stringify({ error: 'Internal server error' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}

export async function GET() {
  return new Response(JSON.stringify({ status: 'ok', model: 'c-code-llm' }), {
    headers: { 'Content-Type': 'application/json' },
  });
}
