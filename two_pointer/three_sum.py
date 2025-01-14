def threesum(nums: list[int]):
    ans = []
    nums.sort()
    for i, a in enumerate(nums):
        if(i > 0 and a == nums[i-1]):
            continue
        low = i + 1
        high = len(nums) - 1
        while(low < high):
            sum = nums[low] + nums[high] + a
            if(sum == 0):
                ans.append([a, nums[low], nums[high]])
                low += 1
                while(nums[low] == nums[low - 1] and low < high):
                    low += 1
            elif(sum > 0):
                high -= 1
            elif(sum < 0):
                low += 1
    return ans
                

nums = [-2,0,0,2,2]
print(threesum(nums))