/*
 * Reverse a singly linked list.
 * head may be NULL. Returns new head.
 */

#include <stdlib.h>

struct Node { int val; struct Node *next; };

struct Node* reverse(struct Node* head) {
    struct Node* prev = NULL;
    struct Node* cur = head;
    while (cur) {
        struct Node* nxt = cur->next;
        cur->next = prev;
        prev = cur;
        cur = nxt;
    }
    return prev;
}
