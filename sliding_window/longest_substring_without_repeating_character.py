def lengthOfLongestSubstring(s: str) -> int:
    left = 0
    charSet = set()
    maxVal = 0
    for right in range(len(s)):
        while s[right] in charSet:
            charSet.remove(s[left])
            left += 1
        charSet.add(s[right])
        maxVal = max(maxVal, right - left + 1)
    return maxVal

s = "dvdf"
print(lengthOfLongestSubstring(s))
            