def dailyTemperatureBrute(temperatures: list[int]) -> list[int]:
    ans = []
    for i in range(len(temperatures)):
        flag = False
        larger = i
        for j in range(i, len(temperatures)):
            if(temperatures[j] > temperatures[i]):
                flag = True
                larger = j
                break
        if flag:
            ans.append(j - i)
        else:
            ans.append(0)
    return ans

def dailyTemperatureOptimal(temperatures: list[int]) -> list[int]:
    ans = [0] * len(temperatures)
    st = [] #pair of index and temp
    
    for i, t in enumerate(temperatures):
        while st and t > st[-1][0]:
            stackT, stackInd = st.pop()
            ans[stackInd] = (i - stackInd)
        st.append([t,i])
    return ans
            
                
            
                

temperatures = [30,38,30,36,35,40,28]
print(dailyTemperatureOptimal(temperatures))
    