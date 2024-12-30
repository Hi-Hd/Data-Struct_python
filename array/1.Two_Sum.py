def TwoSumBrute(nums: list[int], sum: int) -> list[int]:
    array = []
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if(nums[i] + nums[j] == sum):
                print("runs")
                array.append(i)
                array.append(j)
                return array
    
    
def TwoSumOptimal(nums: list[int], sum: int) -> list[int]:
    dic = {}
    ans = []
    for i in range(len(nums)):
        if(sum - nums[i] in dic):
            ans.append(dic[sum - nums[i]])
            ans.append(i)
        dic[nums[i]] = i
    return ans
nums = [5,5]
arr = TwoSumBrute(nums, 111)
print(arr)