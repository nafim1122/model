"""Generate a safe sample JSONL (20 records) for SFT using raw triple-quoted strings.
Writes dataset_tools/retrain_prompts_sample20.jsonl
"""
import json
from pathlib import Path

pairs = [
    ("Write a simple C program to check if a number is prime or not.", r"""#include <stdio.h>
#include <math.h>
int main() {
    int n, i, flag = 1;
    if (scanf("%d", &n) != 1) return 0;
    if (n <= 1) { printf("Not prime\n"); return 0; }
    for (i = 2; i <= (int)sqrt((double)n); ++i) {
        if (n % i == 0) { flag = 0; break; }
    }
    printf(flag ? "Prime\n" : "Not prime\n");
    return 0;
}"""),
    ("How do you reverse a string in C? Give code.", r"""#include <stdio.h>
#include <string.h>
int main() {
    char s[1024];
    if (!fgets(s, sizeof(s), stdin)) return 0;
    size_t n = strlen(s);
    if (n && s[n-1]=='\n') s[--n] = '\0';
    for (size_t i = 0; i < n/2; ++i) {
        char tmp = s[i]; s[i] = s[n-1-i]; s[n-1-i] = tmp;
    }
    printf("%s\n", s);
    return 0;
}"""),
    ("Print Fibonacci numbers in C.", r"""#include <stdio.h>
int main() {
    int n = 10; /* change as needed */
    int a = 0, b = 1;
    for (int i = 0; i < n; ++i) {
        printf("%d ", a);
        int t = a + b; a = b; b = t;
    }
    printf("\n");
    return 0;
}"""),
    ("Find the largest number in an array in C.", r"""#include <stdio.h>
int main(){ int arr[] = {3,7,2,9,4}; int n = sizeof(arr)/sizeof(arr[0]); int max = arr[0]; for (int i=1;i<n;++i) if (arr[i]>max) max=arr[i]; printf("%d\n", max); return 0; }"""),
    ("Count vowels in a string using C.", r"""#include <stdio.h>
#include <ctype.h>
int main(){ char s[1024]; if(!fgets(s,sizeof(s),stdin)) return 0; int c=0; for (char *p=s; *p; ++p){ char ch = tolower((unsigned char)*p); if (ch=='a'||ch=='e'||ch=='i'||ch=='o'||ch=='u') ++c; } printf("%d\n", c); return 0; }"""),
    ("Write a C loop that prints numbers from 1 to 100.", r"""#include <stdio.h>
int main(){ for(int i=1;i<=100;++i) printf("%d\n", i); return 0; }"""),
    ("Sort an array in C (bubble sort).", r"""#include <stdio.h>
void bubble(int *a,int n){ for(int i=0;i<n-1;++i) for(int j=0;j<n-1-i;++j) if(a[j]>a[j+1]){int t=a[j];a[j]=a[j+1];a[j+1]=t;} }
int main(){ int a[]={5,2,9,1,3},n=5; bubble(a,n); for(int i=0;i<n;++i) printf("%d ",a[i]); printf("\n"); return 0; }"""),
    ("Sample code for matrix multiplication in C.", r"""#include <stdio.h>
int main(){ int A[2][3]={{1,2,3},{4,5,6}}; int B[3][2]={{7,8},{9,10},{11,12}}; int C[2][2] = {0}; for(int i=0;i<2;i++) for(int j=0;j<2;j++) for(int k=0;k<3;k++) C[i][j]+=A[i][k]*B[k][j]; for(int i=0;i<2;i++){ for(int j=0;j<2;j++) printf("%d ",C[i][j]); printf("\n"); } return 0; }"""),
    ("Write a C program to check if a number is Armstrong.", r"""#include <stdio.h>
#include <math.h>
int main(){ int n; if(scanf("%d",&n)!=1) return 0; int t=n, sum=0, d, digits=0; int tmp=n; while(tmp){ digits++; tmp/=10; } while(n){ d=n%10; sum += (int)pow(d,digits); n/=10; } printf(sum==t?"Armstrong\n":"Not Armstrong\n"); return 0; }"""),
    ("Calculate factorial using recursion in C.", r"""#include <stdio.h>
long fact(int n){ return n<=1?1:n*fact(n-1); }
int main(){ int n=5; printf("%ld\n", fact(n)); return 0; }"""),
    ("Check palindrome number in C.", r"""#include <stdio.h>
int main(){ int n,rev=0,orig; if(scanf("%d",&n)!=1) return 0; orig=n; while(n){ rev = rev*10 + n%10; n/=10; } printf(rev==orig?"Palindrome\n":"Not palindrome\n"); return 0; }"""),
]

out = Path(__file__).with_name('retrain_prompts_sample20.jsonl')
with out.open('w', encoding='utf-8') as f:
    for p,c in pairs:
        json.dump({'prompt': p, 'completion': c, 'pair_type': 'sft', 'origin': 'generated_sample20'}, f, ensure_ascii=False)
        f.write('\n')
print(f"Wrote {out} with {len(pairs)} records")
