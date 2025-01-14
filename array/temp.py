def encode(strsList):
    st = ""
    for i in strsList:
        cnt = len(i)
        st += str(cnt) + "#" + i
    return st

def decode(strs):
    lis = []
    i = 0
    while i < len(strs):
        j = i
        while(strs[j] != '#'):
            j += 1
        cnt = int(strs[i:j])
        lis.append(strs[j+1:j+1+cnt])
        i = j + 1 + cnt
    return lis
            
        

strsList = ['hello', 'i', 'am', 'harshit','iLoveYouHarshit']
lis = encode(strsList)
print(decode(lis))