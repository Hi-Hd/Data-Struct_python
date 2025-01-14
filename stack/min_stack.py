class MinStack(object):
    
    def __init__(self):
        self.stack = []

    def push(self, val):
        if self.stack:
            minVal = self.stack[-1][1]
            if val < minVal:
                minVal = val
            self.stack.append([val, minVal])
        else:
            self.stack.append([val, val])

    def pop(self):
        if self.stack:
            return self.stack.pop()[0]        

    def top(self):
        if self.stack:
            return self.stack[-1][0]
        

    def getMin(self):
        if self.stack:
            return self.stack[-1][1]


st = MinStack()
st.push(5)
st.push(10)
st.push(15)
st.push(-55)
st.push(11)
print(st.getMin())


        

