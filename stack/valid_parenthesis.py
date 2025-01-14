def validParenthesis(s: str) -> bool:
    st = []
    if(len(s) <= 1):
        return False
    for i in s:
        if(i == '(' or i == "{" or i == "["):
            st.append(i)
        else:
            popped = st.pop()
            print(st)
            if(ord(i) - ord(popped) >2 or ord(i) - ord(popped) < 0):
                return False
    if st:
        return False
    return True

s = "({{{{}}}))"
print(validParenthesis(s))
print(ord(")") - ord("{"))