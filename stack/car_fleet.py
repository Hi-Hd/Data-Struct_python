def carFleet(target: int, position: list[int], speed: list[int]) -> int:
    pair = [[p,s] for p, s in zip(position, speed)]
    stack = []
    for p, s in sorted(pair)[::-1]:
        stack.append((target - p) / s)
        if(len(stack) >= 2 and stack[-1] <= stack[-2]):
            stack.pop()
    return len(stack)
    
    

position = [3,5,7]
speed = [3,2,1]
print(carFleet(10,position, speed))