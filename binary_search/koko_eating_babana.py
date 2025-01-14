import math
def calculateTime(piles: list[int], h: int, k: int) -> bool:
    hours = 0
    for i in piles:
        hours += math.ceil(i / k)
    if(hours <= h):
        return True
    return False

def getMax(piles: list[int]) -> int:
    mx = piles[0]
    for i in piles:
        if(mx < i):
            mx = i
    return mx

def kokoEatingBanana(piles: list[int], h: int) -> int:
    low = 1
    high = getMax(piles)
    while low <= high:
        mid = (low + high) // 2
        if(calculateTime(piles, h, mid)):
            high = mid - 1
        else:
            low = mid + 1
    return low

        

piles = [30,11,23,4,20]
print(kokoEatingBanana(piles, 6))



