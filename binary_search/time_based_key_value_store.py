from collections import defaultdict
class TimeMap:

    def __init__(self):
        self.dic = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.dic[key].append([value, timestamp])
        
    def get(self, key: str, timestamp: int) -> str:
        searchSpace = self.dic[key]
        low = 0
        high = len(searchSpace) - 1
        if searchSpace:
            if timestamp < searchSpace[low][1]:
                return ""
        else:
            return ""
        while low <= high:
            mid = (low + high) // 2
            if(searchSpace[mid][1] == timestamp):
                return searchSpace[mid][0]
            elif(searchSpace[mid][1] > timestamp):
                high = mid - 1
            else:
                low = mid + 1
        return searchSpace[high][0]
        
        

tm = TimeMap()
print(tm.set("a","bar",1))
print(tm.set("x","b",3))
print(tm.get("b",3))
# print(tm.get("love",10))
# print(tm.get("love",15))
# print(tm.get("love", 20))
# print(tm.get("love", 25))