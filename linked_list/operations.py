class Node:
    def __init__(self, data=0, next=None, random = None):
        self.data = data
        self.next = next
        self.random = random

class LinkedList:
    def __init__(self):
        self.head = None
    
    def insert_At_Beginning(self, data):
        newNode = Node(data)
        newNode.next = self.head
        self.head = newNode
    
    def insert_At_End(self, data):
        it = self.head
        if not it:
            self.head = Node(data)
        while it.next != None:
            it = it.next
        newNode = Node(data)
        it.next = newNode
    
    def length(self):
        it = self.head
        cnt = 0
        while it != None:
            cnt += 1
            it = it.next
        print(cnt)
    
    def insert_At_Position(self, position, data):
        it = self.head
        pos = 1
        while it and pos != position - 1:
            it = it.next
            pos += 1
        newNode = Node(data)
        newNode.next = it.next
        it.next = newNode
        
    def arrayToLinkedList(self, arr : list):
        self.head = Node(arr[0])
        it = self.head
        for i in range(1, len(arr)):
            it.next = Node(arr[i])
            it = it.next
    
    def reverseLinkedList(self, node: Node) -> Node:
        if node == None or node.next == None:
            return node
        newHead = self.reverseLinkedList(node.next)
        headNext = node.next
        headNext.next = node
        node.next = None
        return newHead
        
    def reverse(self):
        self.head = self.reverseLinkedList(self.head)
    
    def iterateLinkedList(self):
        it = self.head
        if not it:
            print("linked list is empty:")
            return
        while it != None:
            if it.next == None:
                print(it.data)
                it = it.next
                continue
            print(it.data, end=" -> ")
            it = it.next
        
    def removeNthFromEnd(self, head: Node, n:int) -> Node:
        pt = 0
        slow = head
        while slow and pt != n:
            slow = slow.next
            pt += 1
        fast = head
        if not slow:
            print("works")
            temp = fast
            head = head.next
            temp.next = None
            self.head = head
            return
        while slow.next:
            slow = slow.next
            fast = fast.next
        temp = fast.next
        fast.next = temp.next
        temp.next = None
            
    def removeN(self, n):
        self.removeNthFromEnd(self.head, n)
    
    def iterateLLWithNode(self, node: Node):
        it = node
        while it:
            print(it.data, end = " -> ")
            it = it.next
        
    
    def reorderList(self):
        slow = self.head
        fast = self.head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next
            if fast:
                fast = fast.next
        second = slow.next
        slow.next = None
        prev = None
        
        while second:
            tmp = second.next
            second.next = prev
            prev = second
            second = tmp
        
        #merge the two hales in the list
        first, second = self.head, prev
        while second:
            temp1, temp2 = first.next, second.next
            first.next = second
            second.next = temp1
            first, second = temp1, temp2
    
    def copyRandomList(head: Node) -> Node:
        oldToCopy = {None: None}
        
        cur = head
        while cur:
            copy = Node(cur.data)
            oldToCopy[cur] = copy
            cur = cur.next
        
        cur = head
        while cur:
            copy:Node = oldToCopy[cur]
            copy.next = oldToCopy[cur.next]
            copy.random = oldToCopy[cur.random]
            cur = cur.next
        
        return oldToCopy[head]
    
    def reverseList(self, head: Node) -> Node:
        prevptr = None
        currPtr = head
        while currPtr:
            nextPtr = currPtr.next
            currPtr.next = prevptr
            prevptr = currPtr
            currPtr = nextPtr
        return prevptr
    def mergeTwoLists(self, list1: Node, list2: Node) -> Node:
        dummy = LinkedList()
        tail :Node = dummy
        while list1 and list2:
            if list1.data <= list2.data:
                tail.next = list1
                list1 = list1.next
            else:
                tail.next = list2
                list2 = list2.next
            tail = tail.next
        if list1:
            tail.next = list1
        if list2:
            tail.next = list2
        return dummy.next
    
    def addTwoNumbers(self, l1: Node, l2: Node) -> Node:
        num1 = 0
        bases = 1
        while l1:
            num1 = bases * l1.data + num1
            bases *= 10
            l1 = l1.next
        num2 = 0
        bases = 1
        while l2:
            num2 = bases * l2.data + num2
            bases *= 10
            l2 = l2.next
        ans = num1 + num2
        print(ans)
        head = Node()
        it = head
        lenans = str(ans)
        for i in range(len(lenans)):
            value = ans % 10
            temp = Node(value)
            ans = int(ans / 10)
            it.next = temp
            it = it.next
        self.head = head.next
    def addTwoNumNoSpace(self, l1: Node, l2: Node) -> Node:
        dummy = Node()
        cur = dummy
        
        carry = 0
        while l1 or l2 or carry:
            v1 = l1.val if l1 else 0
            v2 = l2.val if l2 else 0
            
            #new digit
            val = v1 + v2 + carry
            
            carry = val // 10
            val = val % 10
            cur.next = ListNode(val)
            
            car = cur.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        return dummy
    
    def mergeKLists(self, lists: list[Node]) -> Node:
        if not lists or len(lists) == 0:
            return None
        
        while len(lists) > 1:
            mergedList = []
            
            for i in range(0,len(lists), 0):
                list1 = lists[i]
                lists2 = lists[i + 1]
                merged = mergedListMan(list1, lists2)
                mergedList.append(merged)
            lists = mergedList
        return lists[0]
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        dummy = ListNode(0, head)
        groupPrev = dummy
        
        while True:
            kth = self.getKth(groupPrev, k)
            if not kth:
                break
            groupNext = kth.next
            
            #reverse group
            prev, curr = kth.next, groupPrev.next
            while curr != groupNext:
                temp = curr.next
                curr.next = prev
                prev = curr
                curr = temp
            groupPrev.next = kth
            
    def getKth(self, curr, k):
        while curr and k > 0:
            curr = curr.next
            k -= 1
        return curr
        
        
        
            
    
arr1 = [0]
arr2 = [0]
ll1 = LinkedList()
ll1.arrayToLinkedList(arr1)
ll2 = LinkedList()
ll2.arrayToLinkedList(arr2)

ll3 = LinkedList()
ll3.addTwoNumbers(ll1.head, ll2.head)
ll3.iterateLinkedList()
