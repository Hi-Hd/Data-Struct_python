def validAnagramBrute(str1, str2):
    sorted_str1 = ''.join(sorted(str1))
    sorted_str2 = ''.join(sorted(str2))
    if(len(str1) != len(str2)):
        return False
    for i in range(max(len(str1), len(str2))):
        if sorted_str1[i] != sorted_str2[i]:
            return False
    return True

def validAnagramOptimal(s, t):
    count1 = {}
    count2 = {}
    if(len(s) != len(t)):
        return False
    for i in range(len(s)):
        count1[s[i]] = 1 + count1.get(s[i], 0)
        count2[t[i]] = 1 + count2.get(t[i], 0)
        
    #now this is done
    for i in count1:
        if(count1.get(i) != count2.get(i, 0)):
            return False
    return True

def validAnagramNoSpace(s: str, t: str) -> bool:
    val1 = 0
    val2 = 0
    
    if len(s) != len(t):
        return False
    
    for i in range(len(s)):
        val1 += ord(s[i])
        val2 += ord(t[i])
    if val1 == val2:
        return True
    else:
        return False

s = "racecar"
t = "carrace"
print(validAnagramNoSpace(s, t))
    