def seach2D(matrix: list[list[int]], target: int) -> bool:
    outerLow = 0
    outerHigh = len(matrix) - 1
    while(outerLow <= outerHigh):
        outerMid = (outerHigh + outerLow) // 2
        if(matrix[outerMid][0] <= target and matrix[outerMid][len(matrix[outerMid]) - 1] >= target):
            print(outerMid)
            curr = matrix[outerMid]
            low = 0
            high = len(curr) - 1
            while(low <= high):
                mid = (low + high) // 2
                print(low, mid, high)
                if(curr[mid] == target):
                    return True
                elif(curr[mid] > target):
                    high = mid - 1
                else:
                    low = mid + 1
            return False
        elif(matrix[outerMid][0] >= target and matrix[outerMid][len(matrix[outerMid]) - 1] >= target):
            outerHigh = outerMid - 1
        else:
            outerLow = outerMid + 1
    return False

matrix=[[1,3,5,7],[10,11,16,20],[23,30,34,60]]

print(seach2D(matrix, 13))