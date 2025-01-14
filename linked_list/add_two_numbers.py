class ListNode:
    def __init__(self, val = 0, next = None):
        self.val = val
        self.next = next

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = ListNode()
        tail = dummy
        carry = 0
        
        while l1 or l2 or carry:
            num1 = l1.val if l1 else 0
            print(num1)
            num2 = l2.val if l2 else 0
            print(num2)
            
            ans = num1 + num2 + carry
            carry = ans // 10
            ans = ans % 10
            
            tail.next = ListNode(ans)
            tail = tail.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        return dummy.next
            