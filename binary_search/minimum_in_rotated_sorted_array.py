def minimumInRotatedSortedArray(nums: list[int]) -> int:
    l = 0
    h = len(nums) - 1
    smallest = nums[0]
    while l <= h:
        if(nums[l] == nums[h]):
            l += 1
        m = (l + h) // 2
        print(l,m,h)
        if(l <= h and nums[m] >= nums[l]):
            smallest = min(smallest, nums[l])
            l = m + 1
        else:
            smallest = min(smallest, nums[m])
            h = m - 1
    return smallest
            

nums = [1,3,5]
print(minimumInRotatedSortedArray(nums))
                
            
            