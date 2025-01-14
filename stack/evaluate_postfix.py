def evaluatePostfix(tokens: list[str]) -> int:
    ans = []
    for i in tokens:
        if(i.isdecimal() or (len(i) > 1 and (i[0] == "-" and i[1].isdigit()))):
            ans.append(int(i))
        else:
            operand1 = ans.pop()
            operand2 = ans.pop()
            if(i == "+"):
                ans.append(operand2 + operand1)
            elif(i == "-"):
                ans.append(operand2 - operand1)
            elif(i == "*"):
                ans.append(operand2 * operand1)
            elif(i == "/"):
                ans.append(int(operand2 / operand1))
    return ans[0]
    
tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
print(evaluatePostfix(tokens))
