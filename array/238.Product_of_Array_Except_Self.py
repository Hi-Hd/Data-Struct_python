def productExceptSelfBrute(nums):
    pre = [1] * len(nums)
    prod = 1
    for i in range(len(nums)):
        prod *= nums[i]
        pre[i] = prod
    post = [1] * len(nums)
    prod = 1
    for i in range(len(nums) - 1, -1,-1):
        prod *= nums[i]
        post[i] = prod
    ans = [1] * len(nums)
    for i in range(len(ans)):
        if(i == 0):
            temp = 1 * post[i + 1]
            ans[i] = temp
            continue
        if(i == len(ans) - 1):
            temp = 1 * pre[i - 1]
            ans[i] = temp
            continue
        temp = pre[i-1] * post[i+1]
        ans[i] = temp
    return ans

nums = [3,4,7]
print(productExceptSelfBrute(nums))