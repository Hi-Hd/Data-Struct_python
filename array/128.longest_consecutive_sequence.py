def longestConsecutiveSequenceBrute(nums: list[int]) -> int:
    nums.sort()
    cnt = 1
    i = 0
    maxCnt = 0
    if(len(nums) == 1):
        return 1
    while i < len(nums) - 1:
        if(nums[i] + 1 == nums[i+1]):
            cnt += 1
        else:
            cnt = 1
        maxCnt = max(cnt, maxCnt)
        i += 1
    return maxCnt
     

def longestConsecutiveSequenceOptimal(nums: list[int]) -> int:
    numSet = set(nums)
    longest = 0
    
    for n in nums:
        if(n-1) not in numSet:
            length = 0
            while(n + length) in numSet:
                length += 1
            longest = max(length, longest)
    return longest
            
            
    
       
nums =[10, 4, 12, 1, 2, 2, 3]
print(longestConsecutiveSequenceOptimal(nums))