def median(nums1: list[int], nums2: list[int]) -> int:
    A, B = nums1, nums2
    total = len(A) + len(B)
    half = total // 2
    if(len(A) > len(B)):
        A, B = B, A
    l, r = 0, len(A) - 1
    while l <= r:
        i = (l + r) // 2
        j = half - i - 2
        
        Aleft = A[i] if i >= 0 else float("-infinity")
        ARight = A[i + 1] if i + 1 < len(A) else float("infinity")
        Bleft = B[i] if j >= 0 else float("-infinity")
        Bright = B[i + 1] if j + 1 < len(B) else float("infinity")
        if Aleft <= Bright and Bleft <= ARight:
            if total % 2:
                return min(ARight, Bright)
            return (max(Aleft, Bleft) + min(ARight, Bright) / 2)
        elif Aleft > Bright:
            r = i - 1
        else:
            l = i + 1
        
        
print(median([1,2,3,4,5,6,7,8], [1,2,3,4,5]))