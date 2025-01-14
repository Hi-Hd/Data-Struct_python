def trappingRain(height: list[int])->int:
    l = 0
    r = len(height) - 1
    maxL = height[l]
    maxR = height[r]
    totalTrap = 0
    while(l < r):
        if(maxL <= maxR):
            l += 1
            if(maxL - height[l] >= 0):
                totalTrap += maxL - height[l]
            maxL = max(maxL, height[l])
        else:
            r -= 1
            if(maxR - height[r] >= 0):
                totalTrap += maxR - height[r]
            maxR = max(maxR, height[r])
    return totalTrap
            


height = [0,1,0,2,1,0,1,3,2,1,2,1]
print(trappingRain(height))