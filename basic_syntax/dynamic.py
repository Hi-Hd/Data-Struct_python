import math

#dynamically typed
n = 0
print("n = ", n)
n = "bussy"
print("n = ", n)


#multiple assignment of variables
n, m = "hello", 123
print("value of n and m is : ", n, " and m = ", m)

#increment and decrement
n = 10
n += 1
print(n)
n -= 1
print(n)

#null is none in python
print(type(n))
n = None
print(n)
print(type(n))

#if in python

n = 10
if n > 10:
    print("greater then 10")
else:
    print("not greater then 10")
    
if(n > 20):
    pass
elif n > 10:
    pass
else:
    print("not greater then 20 or 10")
    
#multiline condition in if statement
n, m = 1, 2
if((n > 2 and n != m) or n == m):
    n += 1
    
#and = &&
#or = ||
#not = !

#while loop

n = 0
while n < 5:
    print(n)
    n += 1
    
#for loop
for i in range(5,2,-1):
    print(i)
    
#divison
print(int(10/3))
print("\n\n\n\n")

print(math.floor(3/2))
print(math.ceil(3/2))
print(math.sqrt(9))
print(math.pow(2,3))

float("inf")
float("-inf")

arr = [1,2,3,4,5]
arr.append(1)
print(arr)
arr.pop()
print(arr)
arr.insert(1,7)
print(arr)
arr[0] = 10000
print(arr)

n = 10
arr1 = [1] * n
print(arr1)
print(len(arr1))

#slicing an array
print(arr[1:3])

#unpacking an array, make sure that the number of variable in the left match though
a, b, c = [1,2,3] 
print(a)
print(b)
print(c)

print(arr)
for i in range(len(arr)):
    print(arr[i])
    
for i, n in enumerate(arr):
    print(i, " ", n)
    
arr1 = [1,2,3,4]
arr2 = [5,6,7,8]
arr.sort(reverse=True)


arr = ['bob', 'alice', 'jane', 'doe']
arr.sort(key = lambda x: len(x))
for i in arr:
    print(i)
    
arr = [[0] * 4 for i in range(4)]
for i in arr:
    print(i)
    
print(ord('a'))


from collections import deque

queue = deque()
queue.append(1)
queue.append(2)
queue.append(3)
print(queue)
queue.popleft()
print(queue)
queue.appendleft(1)
print(queue)

hash = set()
hash.add(1)
hash.add(2)
hash.add(3)
hash.add(4)
print(hash)
hash.add(1)
print(hash)
