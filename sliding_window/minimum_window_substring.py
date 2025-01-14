def minimumWindowSubstring(s: str, t: str) -> str:
    # hash1 = {}
    # hash2 = {}
    # for i in range(len(t)):
    #     hash2[t[i]] = 1 + hash2.get(t[i], 0)
    # left = 0
    # minStr = s
    # for right in range(len(s)):
    #     hash1[s[right]] = 1 + hash1.get(s[right], 0)
    #     flag = True
    #     for i in hash2:
    #         if(hash2[i] != hash1.get(i, 0)):
    #             flag = False
    #             break
    #     if not flag:
    #         continue
    #     else:
    #         if((right - left + 1) < len(minStr)):
    #             minStr = s[left: right + 1]
    #         while(flag):
    #             hash1[s[left]] -= 1
    #             left += 1
    #             for i in hash2:
    #                 if(hash2[i] != hash1.get(i, 0)):
    #                     flag = False
    #                     break
    #     print(left, right)
    #     print(hash1)
    # print(hash1)
    # print(left)
    
    # return minStr
    if t == "": return ""
    
    countT, window = {}, {}
    for i in range(len(t)):
        countT[t[i]] = 1 + countT.get(t[i], 0)
    have, need = 0, len(t)
    res = [-1,-1]
    resLen = float("infinity")
    left = 0
    for right in range(len(s)):
        c = s[right]
        window[c] = 1 + window.get(c, 0)
        if c in countT and window[c] == countT[c]:
            have += 1
        while have == need:
            if(right - left + 1) < resLen:
                res = [left, right]
                resLen = (right - left + 1)
            window[s[left]] -= 1
            if s[left] in countT and window[s[left]] < countT[s[left]]:
                have -= 1
            left += 1
    l, r = res
    return s[l:r + 1] if resLen != float("infinity") else ""

            
print(minimumWindowSubstring("adobecodebanc", t = "abc"))