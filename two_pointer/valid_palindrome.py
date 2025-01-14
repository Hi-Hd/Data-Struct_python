def requiredConverter(st: str) -> str:
    newSt = ""
    for i in st:
        if((i >= "a" and i <= "z") or (i >= "A" and i <= "Z") or (i >= "0" and i <= "9")):
            newSt += i.lower()
    return newSt


def isPalindrome(s: str) -> bool:
    newSt = requiredConverter(st)
    i = 0
    j = len(st) - 1
    while(i < j):
        if(st[i] != st[j]):
            return False
        i += 1
        j -= 1
    return True

def isAlphaNum(c: chr) -> bool:
    if((c >= "a" and c <= "z") or (c >= "A" and c <= "Z") or (c >= "0" and c <= "9")):
        return True
    else:
        return False

def isPalindromeOptimal(s: str) -> bool:
    i = 0
    j = len(s) - 1
    while i < j:
        while( i < j and not(isAlphaNum(s[i]))):
            i += 1
        while(i < j and not(isAlphaNum(s[j]))):
            j -= 1
        if(s[i].lower() != s[j].lower()):
            return False
        i += 1
        j -= 1
    return True


c = 'tab a cat'
print(isPalindromeOptimal(c))
