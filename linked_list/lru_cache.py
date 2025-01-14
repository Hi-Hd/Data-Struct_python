class Node:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.prev = self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = {} #node to value
        
        #left=LRU right= MRU
        self.left, self.right = Node(0,0), Node(0, 0)
        self.left.next, self.right.prev = self.right, self.left
    
    #remove from left
    def remove(self, node: Node):
        prev, nxt = node.prev, node.next
        prev.next, nxt.prev = nxt, prev
        
    #insert node at right(the last place)
    def insert(self, node : Node):
        lastNode : Node= self.right.prev
        newNode : Node = Node(node.key, node.val)
        self.right.prev = newNode
        lastNode.next = newNode
        newNode.prev = lastNode
        newNode.next = self.right
    
    def get(self, key: int) -> int:
        if key in self.cache:
            self.remove(self.cache[key])
            self.insert(self.cache[key])
            return self.cache[key].val
            
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.remove(self.cache[key])
        self.cache[key] = Node(key, value)
        self.insert(self.cache[key])
        
        if len(self.cache) > self.cap:
            #remove from the list and delete the LRU From the hashmap
            lru = self.left.next
            self.remove(lru)
            del self.cache[lru.key]
