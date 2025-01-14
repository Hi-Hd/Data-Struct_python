class Node:
    def __init__(self, val=0, next = None,random = None):
        self.val = val
        self.next = next
        self.random = random
    
class Solution:
    def copyRandomList(self, head: Node) -> Node:
        mp = {None: None}
        it = dummy
        dummy = Node()
        while it:
            mp[it] = Node(it.val)
            it = it.next
        it = head
        while it:
            mp[it].next = mp[it.next]
            mp[it].random = mp[it.random]
            it = it.next
        return mp[head]