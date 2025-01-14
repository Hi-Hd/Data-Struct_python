def twoSumBrute(nums: list[int], target: int) -> list[int]:
    dic = {}
    for i in range(len(nums)):
        if((target - nums[i]) in dic):
            return [dic[target - nums[i]] + 1, i + 1]
        dic[nums[i]] = i 
      
def twoSumOptimal(nums: list[int], target: int) -> list[int]:
    i = 0
    j = len(nums) - 1
    while(i < j):
        if((nums[i] + nums[j]) == target):
            return [i+1, j+1]
        if((nums[i] + nums[j]) > target):
            j -= 1
        elif((nums[i] + nums[j]) < target):
            i += 1
            
nums = [2,3,4]
print(twoSumOptimal(nums, 6))