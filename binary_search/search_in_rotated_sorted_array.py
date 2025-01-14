def searchInRotatedSortedArray(nums: list[int], target: int):
    l = 0
    r = len(nums) - 1
    while l <= r:
        m = (l + r) // 2
        if(nums[m] == target):
            return m
        if(nums[m] >= nums[l]):
            #left is sorted
            if(nums[m] >= target and target >= nums[l]):
                r = m - 1
            else:
                l = m + 1
        else:
            #right is sorted
            if(nums[r] >= target and target >= nums[m]):
                l = m + 1
            else:
                r = m - 1
    return -1


nums = [3,4,5,6,1,2]
print(searchInRotatedSortedArray(nums, 1))
                