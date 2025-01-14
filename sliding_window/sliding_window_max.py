from collections import deque
def slidingWindowMaximum(nums: list[int], k : int) -> int:
    out = []
    l = r = 0
    q = deque() #index
    while r < len(nums):
        while q and nums[q[-1]] < nums[r]:
            q.pop()
        q.append(r)
        
        #remove value from window
        if l > q[0]:
            print("work")
            q.popleft()
        if (r + 1) >= k:
            out.append(nums[q[0]])
            l += 1
        r += 1
    return out
        
    

nums = [1,3,-1,-3,5,3,6,7]
print(slidingWindowMaximum(nums, 3))