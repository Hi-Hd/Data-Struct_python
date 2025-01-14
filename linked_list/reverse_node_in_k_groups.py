class ListNode:
    def __init__(self, val = 0, next = None):
        self.val = val
        self.next = next

def reverseKGroup(head: ListNode, k: int) -> ListNode:
    def reverse(head: ListNode, k: int) -> ListNode:
        prev, curr = None, head
        while k:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
            k-= 1
        return prev
    count = 0
    node = head
    while node:
        count += 1
        node = node.next
    dummy = ListNode(0, head)
    prev_group_end = dummy
    while count >= k:
        group_start = prev_group_end.next
        group_end = group_start
        for i in range(k - 1):
            group_end = group_end.next
        next_group_start = group_end.next
        group_end.next = None
        
        #reverse the current group
        prev_group_end.next = reverse(group_start, k)
        group_start.next = next_group_start
        prev_group_end = group_start
        
        count -= k
    return dummy.next
    
            