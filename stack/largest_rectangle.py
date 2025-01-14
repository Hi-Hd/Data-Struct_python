def largestRectangle(heights: int[list]) -> int:
    st = [] #pair (index, height)
    maxArea = 0
    for i, h in enumerate(heights):
        start = i
        while st and st[-1][1] > h:
            index, height = st.pop()
            maxArea = max(maxArea, height * (i - index))
            start = index
        st.append((start, h))
        
    for i, h in enumerate(st):
        maxArea = max(maxArea, h * (len(heights) - i))
    return maxArea
