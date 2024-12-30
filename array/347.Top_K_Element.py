def topKElement(nums : list[int], k: int) -> list[int]:
    count  = {}
    freq = [[] for i in range(len(nums) + 1)]
    for n in nums:
        count[n] = 1 + count.get(n, 0)
    for n, c in count.items():
        freq[q].append(n)
    res = []
    for i in range(len(freq) - 1, 0, -1):
        for n in freq[i]:
            res.append(n)
            if len(res) == k:
                return res
    
            
    

nums = [1,1,1,2,2,3,5,6,5,3,4,3,2,3,2]
topKElement(nums,2)