def encode(strs: list[str])->str:
    st = ""
    for i in strs:
        temp = str(len(i)) + "#"
        st += temp
        st += i
    return st

def decode(strs)->list[str]:
    ans = []
    i = 0
    temp = ""
    while i < len(strs):
        j = i
        while strs[j] != '#':
            j += 1
        length = int(strs[i:j])
        ans.append(strs[j + 1: j + 1 + length])
        i = j + 1 + length
    return ans
        
        
        
                
            
            

lis = ["neet","code","love","you"]
st = "4#neet4#code4#love3#you"
print(encode(lis))
print(decode(st))