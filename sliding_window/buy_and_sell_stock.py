def buyAndSellStock(preices: list[int]) -> int:
    sell = 1
    buy = 0
    maxProfit = 0
    while sell < len(preices) and buy < len(preices):
        profit = preices[sell] - preices[buy]
        if(profit > 0):
            maxProfit = max(maxProfit, profit)
            sell += 1
            continue
        else:
            buy = sell
            sell += 1
    return maxProfit

prices = [10,8,7,5,2]
print(buyAndSellStock(prices))