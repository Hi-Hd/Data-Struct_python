from collections import defaultdict
from math import sqrt
import math
from collections import deque

class Node:
    def __init__(self, val=0,next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
    def makeList(self, arr: list[int]):
        head = Node(arr[0])
        it = head
        for i in range(1, len(arr)):
            it.next = Node(arr[i])
            it = it.next
        self.head = head
        return head
    def iterate(self, head : Node):
        it = head
        while it:
            if not it.next:
                print(it.val)
                it = it.next
                continue
            print(it.val, end = " -> ")
            it = it.next
            
    

def validSudoku(board : list[list[str]]) -> bool:
    rows = defaultdict(set)
    cols = defaultdict(set)
    box = defaultdict(set)
    for r in range(9):
        for c in range(9):
            element = board[r][c]
            if element == ".":
                continue
            if(element != "." and element in rows[r] or element in cols[c] or element in box[(r//3, c//3)]):
                return False
            rows[r].add(element)
            cols[c].add(element)
            box[(r//3, c//3)].add(element)
    return True
    
def encode(strs: list[str]) -> str:
    st = []
    for i in strs:
        lengthOfStr = len(i)
        st.append(str(lengthOfStr))
        st.append("#")
        st.append(i)
    return "".join(st)

def decode(s: str) -> list[str]:
    i = 0
    st = []
    while i < len(s):
        j = i + 1
        while(s[j] != "#"):
            j += 1
        lengthOfStr = int(s[i:j])
        st.append(s[j + 1: j + 1 + lengthOfStr])
        i = j + 1 + lengthOfStr
    return st
        
def carFleet(target: int, position: list[int], speed: list[int]) -> int:
    combined = [[p, s] for p,s in zip(position, speed)]
    st = []
    
    for i in sorted(combined, reverse=True):
        timeReq = (target - i[0]) / i[1]
        print(timeReq)
        st.append([i, timeReq])
        if(len(st) > 1 and st[-1][1] <= st[-2][1]):
            st.pop()
    return len(st)

def minimumInSortedArray(nums: list[int]) -> int:
    low = 0
    high = len(nums) - 1
    minimum = nums[0]
    while low <= high:
        mid = (low + high) // 2
        if(nums[low] <= nums[mid]):
            #left sorted array
            minimum = min(minimum, nums[low])
            low = mid + 1
        else:
            #in right sorted
            minimum = min(minimum, nums[mid])
            high = mid
    return minimum

def searchInRotatedArray(nums: list[int], target: int) -> int:
    low = 0
    high = len(nums) - 1
    while low <= high:
        mid = (low + high) // 2
        if(nums[mid] == target):
            return mid
        
        if(nums[low] <= nums[mid]):
            if(target >=nums[low] and target <= nums[mid]):
                high = mid - 1
            else:
                low = mid + 1
        else:
            if(target >= nums[mid] and target <= nums[high]):
                low = mid + 1
            else:
                high = mid - 1
    return -1


def generateParathensis(n: int) -> list[str]:
    s = []
    strs = []
    
    def recurs(open: int, close: int):
        if(open == n and close == n):
            strs.append("".join(s))
            return
        if(open < n):
            s.append("(")
            recurs(open + 1, close)
            s.pop()
        if(close < open):
            s.append(")")
            recurs(open, close + 1)
            s.pop()
    recurs(0, 0)
    return strs

def findMedianSortedArrays(nums1, nums2):
        p1 = 0
        p2 = 0
        st = []
        while(p1 != len(nums1) and p2 != len(nums2)):
            if(nums1[p1] <= nums2[p2]):
                st.append(nums1[p1])
                p1 += 1
            else:
                st.append(nums2[p2])
                p2 += 1
        while(p1 != len(nums1)):
            st.append(nums1[p1])
            p1 += 1
        while(p2 != len(nums2)):
            st.append(nums2[p2])
            p2 += 1
        lengthOfMerged = len(st)
        if(lengthOfMerged % 2 == 0):
            return (st[(lengthOfMerged // 2) - 1] + st[lengthOfMerged // 2]) / 2
        else:
            return st[lengthOfMerged // 2]

def genPara(n: int) -> list[str]:
    s = []
    strs = []
    
    def recur(open: int, close: int):
        if(open == n and close == n):
            strs.append("".join(s))
            return
        if(open < n):
            s.append("(")
            recur(open + 1, close)
            s.pop()
        if(close < open):
            s.append(")")
            recur(open, close + 1)
            s.pop()
    recur(0,0)
    return strs


def trappingRainWater(height: list[int]) -> int:
    sum = 0
    Lpt = 0
    Rpt = len(height) - 1
    LMax = height[Lpt]
    RMax = height[Rpt]
    while(Lpt <= Rpt):
        if(LMax <= RMax):
            sum += min(LMax, RMax) - height[Lpt] if min(LMax, RMax) > height[Lpt] else 0
            LMax = max(LMax, height[Lpt])
            Lpt += 1
        else:
            sum += min(LMax, RMax) - height[Rpt] if min(LMax, RMax) > height[Rpt] else 0
            RMax = max(height[Rpt], RMax)
            Rpt -= 1
    return sum

def longestConsecutiveSequence(nums: list[int]) -> int:
    it = set(nums)
    maxCnt = 0
    for i in nums:
        tmp = i
        while(tmp - 1 in it):
            tmp -= 1
        cnt = 1
        while(tmp + 1 in it):
            tmp += 1
            cnt += 1
        maxCnt = max(cnt, maxCnt)
    return maxCnt

def largestHistogram(heights: list[int]) -> int:
    stack = []
    maxArea = 0
    for i, a in enumerate(heights):
        if not stack or a > stack[-1][1]:
            stack.append([i, a])
        if(a <= stack[-1][1]):
            while stack and stack[-1][1] >= a:
                popped = stack.pop()
                area = (i - popped[0]) * popped[1]
                maxArea = max(area, maxArea)
            stack.append([popped[0], a])
            
    for i in stack:
        area = (len(heights) - i[0]) * i[1]
        maxArea = max(maxArea, area)
    return maxArea
    
def validAnagram(s: str, t: str) -> bool:
    if(len(s) != len(t)):
        return False
    dicS = {}
    dicT = {}
    for i in range(len(s)):
        dicS[s[i]] = 1 + dicS.get(s[i], 0)
        dicT[t[i]] = 1 + dicT.get(t[i], 0)
    for i in dicS:
        if(dicS[i] != dicT.get(i, 0)):
            return False
    return True    

def bestTimeToBuyOrSellStock(prices: list[int]) -> int:
    buy = 0
    maxProfit = 0
    for sell in range(len(prices)):
        profit = prices[sell] - prices[buy]
        if profit >= 0:
            maxProfit = max(maxProfit, profit)
        else:
            buy = sell
    return maxProfit

class TimeMap:
    def __init__(self):
        self.dic = defaultdict(list)
    def set(self, key: str, value: str, timestamp: int) -> None:
        self.dic[key].append([value, timestamp])
    
    def get(self, key: str, timestamp: int) -> str:
        searchSpace = self.dic[key]
        if not searchSpace:
            return ""
        if(searchSpace[0][1] > timestamp):
            return ""
        low = 0
        high = len(searchSpace) - 1
        mid = (low + high) // 2
        while low <= high:
            mid = (low + high) // 2
            if(searchSpace[mid][1] == timestamp):
                return searchSpace[mid][0]
            elif(searchSpace[mid][1] > timestamp):
                high = mid - 1
            else:
                low = mid + 1
        return searchSpace[high][0]
        
def findMedianSortedArrays(nums1: list[int], nums2: list[int]) -> float:
        A, B = nums1, nums2
        total = len(nums1) + len(nums2)
        half = total // 2

        if len(B) < len(A):
            A, B = B, A

        l, r = 0, len(A) - 1
        while True:
            i = (l + r) // 2
            j = half - i - 2

            Aleft = A[i] if i >= 0 else float("-infinity")
            Aright = A[i + 1] if (i + 1) < len(A) else float("infinity")
            Bleft = B[j] if j >= 0 else float("-infinity")
            Bright = B[j + 1] if (j + 1) < len(B) else float("infinity")

            if Aleft <= Bright and Bleft <= Aright:
                if total % 2:
                    return min(Aright, Bright)
                return (max(Aleft, Bleft) + min(Aright, Bright)) / 2
            elif Aleft > Bright:
                r = i - 1
            else:
                l = i + 1
                
                
def medianSorted(nums1: list[int], nums2: list[int])->float:
    A,B = nums1, nums2
    total= len(A) + len(B)
    half = total // 2
    if(len(A) > len(B)):
        A,B = B,A
    
    l = 0
    r = len(A) - 1
    while True:
        i = (l + r) // 2
        j = half - i - 2
        
        Aleft = A[i] if i >= 0 else float("-infinity")
        Aright = A[i + 1] if (i + 1) < len(A) else float("infinity")
        Bleft = B[j] if j >= 0 else float("-infinity")
        Bright = B[j + 1] if (j + 1) < len(B) else float("infinity")

        if(Aleft <= Bright and Bleft <= Aright):
            if total % 2 == 0:
                return ((max(Aleft, Bleft) + min(Aright, Bright)) / 2)
            else:
                return min(Aright, Bright)
        elif(Aleft > Bright):
            r = i - 1
        else:
            l = i + 1
        
def longestSubstringWithoutRepeatingCharacters(s: str) -> int:
    st = set()
    left = 0
    maxLen = 0
    for right in range(len(s)):
        while(s[right] in st):
            st.remove(s[left])
            left += 1
        st.add(s[right])
        print(st)
        maxLen = max(maxLen, right - left  + 1)
    return maxLen

def longestRepeatingCharacterReplacement(s: str, k: int) -> int:
    maxLen = 0
    dic = defaultdict(int)
    left = 0
    for right in range(len(s)):
        dic[s[right]] += 1
        largest = 0
        for i in dic:
            largest = max(largest, dic[i])
        while((right - left + 1) - largest > k):
            dic[s[left]] -= 1
            left += 1
            for i in dic:
                largest = max(largest, dic[i])
        maxLen = max(maxLen, (right - left + 1))
    return maxLen
            
def longestRepeatingCharacterReplacementOptimal(s: str, k: int) -> int:
    left = 0
    maxF = 0
    dic = defaultdict(int)
    maxLen = 0
    for right in range(len(s)):
        dic[s[right]] += 1
        maxF = max(maxF, dic[s[right]])
        
        while((right - left + 1) - maxF > k):
            dic[s[left]] -= 1
            left += 1
            
        maxLen = max(maxLen, (right - left + 1))
    return maxLen
        
def permutationInString(s1: str, s2: str) -> bool:
    if(len(s1) > len(s2)):
        return False
    dic1 = {}
    dic2 = {}
    for i in range(len(s1)):
        dic1[s1[i]] = 1 + dic1.get(s1[i], 0)
        dic2[s2[i]] = 1 + dic2.get(s2[i], 0)
    left = 0
    for right in range(len(s1), len(s2)):
        flag = True
        for i in dic1:
            if(dic1[i] != dic2.get(i, 0)):
                flag = False
                break
        if not flag:
            dic2[s2[left]] -= 1
            left += 1
            dic2[s2[right]] = 1 + dic2.get(s2[right], 0)
        else:
            return True
    flag = True
    for i in dic1:
        if(dic1[i] != dic2.get(i, 0)):
            flag = False
            break
    if flag:
        return True
    return False
    
        
def permutationInStringOptimised(s1: str, s2: str) -> bool:
    count1 = [0] * 26
    count2 = [0] * 26
    if(len(s1) > len(s2)):
        return False
    for i in range(len(s1)):
        count1[(ord(s1[i]) - ord("a"))] += 1
        count2[(ord(s2[i]) - ord("a"))] += 1
    left = 0
    match = 0
    for i in range(26):
        if(count1[i] == count2[i]):
            match += 1
    if(match == 26):
        return True
    for right in range(len(s1), len(s2)):
        if(match == 26):
            return True
        index = ord(s2[right]) - ord("a")
        count2[index] += 1
        if count1[index] == count2[index]:
            match += 1
        elif(count1[index] + 1 == count2[index]):
            match -= 1
        
        index = ord(s2[left]) - ord("a")
        count2[index] -= 1
        if count1[index] == count2[index]:
            match += 1
        elif(count1[index] - 1 == count2[index]):
            match -= 1
        left += 1
    if(match == 26):
        return True
    return False
        

def twoSum1(nums: list[int], target: int) -> list[int]:
    dic = {}
    for i in range(len(nums)):
        if(target - nums[i] in dic):
            return [dic[target - nums[i]], i]
        dic[nums[i]] = i
    return [-1,-1]
   
def productOfArrayExceptSelf(nums: list[int]):
    prefixProd = [1] * len(nums)
    postfixProd = [1] * len(nums)
    prod = 1
    for i in range(len(nums)):
        prod *= nums[i]
        prefixProd[i] = prod
    prod = 1
    for i in range(len(nums) - 1, -1, -1):
        prod *= nums[i]
        postfixProd[i] = prod
    out = [1] * len(nums)
    for i in range(len(out)):
        if(i == 0):
            prod = postfixProd[i + 1] * 1
            out[i] = prod
            continue
        if(i == len(out) - 1):
            prod = prefixProd[i - 1] * 1
            out[i] = prod
            continue
        prod = postfixProd[i + 1] * prefixProd[i - 1]
        out[i] = prod
    return out
        
def containerWithMostWater(heights: list[int]) -> int:
    l = 0
    r = len(heights) - 1
    maxHt = 0
    while l < r:
        ht = (r - l) * min(heights[r], heights[l])
        maxHt = max(maxHt, ht)
        if(heights[l] <= heights[r]):
            l += 1
        else:
            r -= 1
    return maxHt

def longestRepeatingRepeatingDigits(s:str, k:int) -> int:
    hash = {}
    highF = 0
    l = 0
    maxLen = 0
    for r in range(len(s)):
        hash[s[r]] = 1 + hash.get(s[r], 0)
        highF = max(highF, hash[s[r]])
        
        while((r - l + 1) - highF > 0):
            hash[s[l]] -= 1
            l += 1
        
        maxLen = max(maxLen, (r - l + 1))
    return maxLen

def largestAreaInHistogram(heights: list[int]) -> int:
    st = [] #[index, value]
    maxArea = 0
    for i in range(len(heights)):
        index = i
        while(st and st[-1][1] >= heights[i]):
            popped = st.pop()
            area = (i - popped[0]) * popped[1]
            maxArea = max(area, maxArea)
            index = popped[0]
        st.append([index, heights[i]])
    for i in range(len(st)):
        popped = st.pop()
        area = (len(heights) - popped[0]) * popped[1]
        maxArea = max(maxArea, area)
    return maxArea
    

def isAlphaNum(c: chr) -> bool:
    return (c >= "A" and c <= "Z") or (c >= 'a' and c <= 'z') or (c >= "1" and c <= "9")
    


def validPalindrome1(s: str) -> bool:
    i = 0
    j = len(s) - 1
    while i < j:
        while(i < j and not(isAlphaNum(s[i]))):
            i += 1
        while(i < j and not(isAlphaNum(s[j]))):
            j -= 1
        if(s[i].lower() != s[j].lower()):
            return False
        i += 1
        j -= 1
    return True
  
def search2DMatrix(matrix: list[list[int]], target: int) -> bool:
    low = 0
    high = len(matrix) - 1
    while low <= high:
        mid = (low + high) // 2
        currRow = matrix[mid]
        if(currRow[0] <= target and currRow[-1] >= target):
            print("inn here")
            innlow = 0
            innhigh = len(currRow) - 1
            while innlow <= innhigh:
                innmid = (innhigh + innlow) // 2
                if currRow[innmid] == target:
                    return True
                elif(currRow[innmid] > target):
                    innhigh = innmid - 1
                else:
                    innlow = innmid + 1
            return False
        elif(currRow[0] <= target and currRow[-1] <= target):
            low = mid + 1
        else:
            high = mid - 1
    return False

      
def permutationInString(s1: str, s2: str) -> bool:
    if(len(s1) > len(s2)):
        return False
    hash1 = [0] * 26
    hash2 = [0] * 26
    for i in range(len(s1)):
        hash1[(ord(s1[i]) - ord("a"))] += 1
        hash2[(ord(s2[i]) - ord("a"))] += 1
    print(hash1)
    print(hash2)
    match = 0
    for i in range(26):
        if(hash1[i] == hash2[i]):
            match += 1
    l = 0
    for r in range(len(s1), len(s2)):
        print(match)
        print(hash2)
        if match == 26:
            return True
        index = (ord(s2[r]) - ord("a"))
        hash2[index] += 1
        if hash1[index] == hash2[index]:
            match += 1
        elif(hash1[index] + 1) == hash2[index]:
            match -= 1
        
        index = (ord(s2[l]) - ord("a"))
        hash2[index] -= 1
        if hash1[index] == hash2[index]:
            match += 1
        elif(hash1[index] - 1) == hash2[index]:
            match -= 1
        l += 1
    print(match)
    if match == 26:
        return True
    return False

def threeSum(nums: list[int]) -> list[int]:
    nums.sort()
    ans = []
    for i in range(len(nums)):
        if(i > 0 and nums[i] == nums[i - 1]):
            continue
        l = i + 1
        r = len(nums) - 1
        while l < r:
            target = nums[i] + nums[l] + nums[r]
            if target > 0:
                r -= 1
            elif(target < 0):
                l += 1
            else:
                ans.append([nums[i], nums[l], nums[r]])
                l += 1
                while(l < r) and (nums[l] == nums[l - 1]):
                    l += 1
    return ans
                

def evalReversePolish(tokes: list[str]) -> int:
    st = []
    for i in range(len(tokes)):
        if(tokes[i].isdigit() or (len(tokes[i]) > 1 and tokes[i][1].isdigit())):
            st.append(int(tokes[i]))
        else:
            if tokes[i] == "+":
                opr1 = st.pop()
                opr2 = st.pop()
                ans = opr2 + opr1
                st.append(ans)
            elif(tokes[i] == "-"):
                opr1 = st.pop()
                opr2 = st.pop()
                ans = opr2 - opr1
                st.append(ans)
            elif(tokes[i] == "*"):
                opr1 = st.pop()
                opr2 = st.pop()
                ans = opr2 * opr1
                st.append(ans)
            else:
                opr1 = st.pop()
                opr2 = st.pop()
                ans = int(opr2 / opr1)
                st.append(ans)
    return st[0]
            
def medianOfSortedList(nums1: list[int], nums2: list[int]) -> int:
    A, B = nums1, nums2
    if len(A) > len(B):
        A, B = B, A
    total = len(A) + len(B)
    half = total // 2
    low = 0
    high = len(A) - 1
    while True:
        midA = (low + high) // 2
        midB = half - midA - 2
        Aleft = A[midA] if midA >= 0 else float("-inf")
        Aright = A[midA + 1] if midA + 1 < len(A) else float("inf")
        Bleft = B[midB] if midB >= 0 else float("-inf")
        Bright = B[midB + 1] if (midB + 1) < len(B) else float("inf")
        
        if(Aleft <= Bright and Bleft <= Aright):
            if total % 2:
                #odd
                return min(Aright, Bright)
            else:
                return (((max(Aleft, Bleft) + min(Aright, Bright)) / 2))
        elif(Aleft > Bright):
            high = midA - 1
        else:
            low = midA + 1


def validParenthesis(s: str):
    st = []
    for i in s:
        if(i == "(" or i == "[" or i == "{"):
            st.append(i)
        else:
            if not st:
                return False
            popped = st.pop()
            if not ((popped == "(" and i == ")") or (popped == "{" and i == "}") or (popped == "[" and i == "]")):
                return False
    if st:
        return False
    return True

def twoIntegerSum2(numbers: list[int], target: int) -> list[int]:
    l = 0
    r = len(numbers) - 1
    while l < r:
        sum = numbers[l] + numbers[r]
        if sum > target:
            r -= 1
        elif sum < target:
            l += 1
        else:
            return [l + 1, r + 1]
    return [-1, -1]

def kokoChecker(piles:list[int], h: int, k:int):
    hours = 0
    for i in piles:
        hours += math.ceil(i / k)
    if hours <= h:
        return True
    return False

def kokoEatingBanana(piles: list[int], h: int) -> int:
    low = 1
    high = max(piles)
    ans = 0
    while(low  <=  high):
        mid = (low + high) // 2
        if(kokoChecker(piles, h, mid)):
            ans = mid
            high = mid - 1
        else:
            low = mid + 1
    return ans

def lengthOfLongestSubstring(s: str):
    if len(s) == 0:
        return 0
    st = set()
    l = 0
    maxLen = 1
    for r in range(len(s)):
        while s[r] in st:
            st.remove(s[l])
            l += 1
        st.add(s[r])
        maxLen = max(maxLen, (r - l + 1))
    return maxLen

def dailyTemp(temperatures: list[int]) -> list[int]:
    ans = [0] * len(temperatures)
    st = [] #[temp, ind]
    for i, a in enumerate(temperatures):
        while(st and a > st[-1][0]):
            popped = st.pop()
            ans[popped[1]] = (i - popped[1])
        st.append([a,i])
    return ans

def carFleet(position: list[int], speed: list[int], target: int) -> int:
    st = []
    combined = [[p,s] for p, s in zip(position, speed)]
    for i in sorted(combined, reverse=True):
        if st:
            elementTime = (target - i[0]) / i[1]
            stackTime = (target - st[-1][0]) / st[-1][1]
            if elementTime > stackTime:
                st.append(i)
            continue
        st.append(i)
    return len(st)
        

def topKFrequent(nums: list[int], k: int) -> list[int]:
    hash = {}
    for i, a in enumerate(nums):
        hash[a] = 1 + hash.get(a, 0)
    print(hash)
    freq = [[] for i in range(len(nums) + 1)]
    for i in hash:
        print(i)
        freq[hash[i]].append(i)
    print(freq)
    out = []
    for i in range(len(freq) - 1, 0, -1):
        for j in freq[i]:
            if(len(out) == k):
                break
            out.append(j)
        if(len(out) == k):
            break
    return out 
        
def minWindow(s: str, t: str) -> str:
    if t == "": return ""
    countT, window = {}, {}
    for c in t:
        countT[c] = 1 + countT.get(c, 0)
    have, need = 0, len(countT)
    res, resLen = [-1,-1], float("infinity")
    l = 0
    for r in range(len(s)):
        c = s[r]
        window[c] = 1 + window.get(c, 0)
        if c in countT and window[c] == countT[c]:
            have += 1
        while have == need:
            if(r - l + 1) < resLen:
                res = [l, r]
                resLen = (r - l + 1)
            window[s[l]] -= 1
            if s[l] in countT and window[s[l]] < countT[s[l]]:
                have -= 1
            l += 1
    if resLen == float("infinity"):
        return ""
    l, r = res
    return s[l: r + 1]
                
            
def minWindowSubstring(s: str, t: str) -> str:
    window = {}
    countT = {}
    for i in t:
        countT[i] = 1 + countT.get(i, 0)
    print(countT)
    have = 0
    need = len(countT)
    l = 0
    res = [-1,-1]
    resLen = float("inf")
    for r in range(len(s)):
        c = s[r]
        window[c] = 1 + window.get(c, 0)
        if c in countT and window[c] == countT[c]:
            have += 1
        while have == need:
            if(r - l + 1) < resLen:
                res = [l, r]
                resLen = r - l + 1
            window[s[l]] -= 1
            if s[l] in countT and window[s[l]] < countT[s[l]]:
                have -= 1
            l += 1
    if resLen == float("inf"):
        return ""
    l, r = res
    return s[l:r+1]
    

from collections import deque

def slidingWindowMax(nums: list[int], k: int) -> list[int]:
    ans = []
    l = r = 0
    q = deque()
    
    while r < len(nums):
        while q and nums[q[-1]] < nums[r]:
            q.pop()
        q.append(r)
        
        if l > q[0]:
            q.popleft()
        
        if r - l + 1 == k:
            ans.append(nums[q[0]])
            l += 1
        r += 1
    return ans


def dailyTemperatures(temperatures: list[int]) -> list[int]:
    ans = [0] * len(temperatures)
    st = [] # [index, temp]
    for temp in range(len(temperatures)):
        while st and st[-1][1] < temperatures[temp]:
            popped = st.pop()
            ans[popped[0]] = temp - popped[0]
        st.append([temp, temperatures[temp]])
    return ans

def findMin(nums: list[int]) -> int:
    l = 0
    h = len(nums) - 1
    ans = float("inf")
    while l <= h:
        m = (l + h) // 2
        if nums[l] <= nums[m]:
            ans = min(ans, nums[l])
            l = m + 1
        else:
            ans = min(ans, nums[m])
            h = m
    return ans


def binarySearch(nums: list[int], target: int) -> int:
    low = 0
    high = len(nums) - 1
    while low <= high:
        mid = (low + high) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] > target:
            high = mid - 1
        else:
            low = mid + 1
            
    return -1

def genPara(n: int) -> list[str]:
    ans = []
    temp = []
    def recur(open: int, close: int):
        if open == n and close == n:
            ans.append("".join(temp))
        if open < n:
            temp.append("(")
            recur(open + 1, close)
            temp.pop()
        if close < open:
            temp.append(")")
            recur(open, close + 1)
            temp.pop()
    recur(0, 0)
    return ans

def searchInRotatedSortedArray(nums: list[int], target: int) -> int:
    l = 0
    h = len(nums) - 1
    while l <= h:
        m = (l + h) // 2
        if nums[m] == target:
            return m
        elif nums[l] <= nums[m]:
            if(target >= nums[l] and target <= nums[m]):
                h = m - 1
            else:
                l = m + 1
        else:
            if(target >= nums[m] and target <= nums[h]):
                l = m + 1
            else:
                h = m - 1
    return -1
        

def ThreeSum1(nums: list[int]) -> int:
    ans = []
    i = 0
    nums.sort()
    while i < len(nums):
        if i > 0 and nums[i] == nums[i - 1]:
            
            i += 1
            continue
        l = i + 1
        h = len(nums) - 1
        while l < h:
            target = nums[i] + nums[l] + nums[h]
            if target > 0:
                h -= 1
            elif target < 0:
                l += 1
            else:
                ans.append([nums[i], nums[l], nums[h]])
                l += 1
                while l < h and nums[l] == nums[l - 1]:
                    l += 1
        i += 1
    return ans
                    
def slidingMax(nums: list[int], k: int) -> list[int]:
    ans = []
    l = 0
    r = 0
    q = deque()
    while r < len(nums):
        while q and nums[q[-1]] < nums[r]:
            q.pop()
        q.append(r)
        if q[0] < l:
            q.popleft()
        if (r - l + 1) == k:
            print(q)
            ans.append(nums[q[0]])
            l += 1
        r += 1
    return ans
            

def largestRectangleInHistogram(heights: list[int]) -> int:
    stack = []
    maxArea = 0
    for i, a in enumerate(heights):
        if not stack or a > stack[-1][1]:
            stack.append([i, a])
        if(a <= stack[-1][1]):
            while stack and stack[-1][1] >= a:
                popped = stack.pop()
                area = (i - popped[0]) * popped[1]
                maxArea = max(area, maxArea)
            stack.append([popped[0], a])
                
    for i in stack:
        area = (len(heights) - i[0]) * i[1]
        maxArea = max(maxArea, area)
    return maxArea


def removeNodeFromEnd(head: Node, n: int) -> Node:
    slow = head
    cnt = 0
    # temp = 
    while slow and cnt != n:
        
        slow = slow.next
        cnt += 1
    newSlow = head
    if slow == None:
        return head.next
    while slow.next:
        slow = slow.next
        newSlow = newSlow.next
    newSlow.next = newSlow.next.next
    return head


def merge(head1: Node, head2: Node) -> Node:
    dummy = Node()
    tail = dummy
    while head1 and head2:
        if head1.val <= head2.val:
            tail.next = head1
            head1 = head1.next
        else:
            tail.next = head2
            head2 = head2.next
        tail = tail.next
    if head1:
        tail.next = head1
    if head2:
        tail.next = head2
    return dummy.next


def mergeKSorted(lists: list[int]):
    if lists == "" or len(lists) < 1:
        return None
    
    while len(lists) > 1:
        mergedList = []
        for i in range(0,len(lists), 2):
            l1 = lists[i]
            l2 = lists[i + 1] if i + 1 < len(lists) else None
            mergedNode = merge(l1, l2)
            mergedList.append(mergedNode)
        list = mergedList
    return list[0]
            
def isAlp(c: chr) -> bool:
    if((c >= "a" and c <= "z") or (c >= "A" and c <= "Z") or (c >= '0' and c <= "9")):
        return True
    return False

def pali(s: str) -> bool:
    i = 0
    j = len(s) - 1
    while i <= j:
        while i < j and not isAlp(s[i]):
            i += 1
        while i < j and not isAlp(s[j]):
            j -= 1
        if s[i].lower() != s[j].lower():
            print(s[i])
            print(s[j])
            return False
        i += 1
        j -= 1
    return True


def groupAnagram(strs: list[str]) -> list[str]:
    dic = defaultdict(list)
    for s in strs:
        count = [0] * 26
        for c in s:
            count[ord(c) - ord("a")] += 1
        dic[tuple(count)].append(s)
    return dic.values()
    

def mergeList(l1: Node, l2: Node)->Node:
    dummy = Node()
    tail = dummy
    while l1 and l2:
        if l1.val<=l2.val:
            tail.next = l1
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        tail = tail.next
    if l1:
        tail.next = l1
    if l2:
        tail.next = l2
    return dummy.next


def mergeKsortedLinkedList(lists: list[Node]) -> Node:
    if len(lists) < 1 or lists == "":
        return None
    
    while len(list) > 1:
        mergedList = []
        for l in range(0,len(lists), 2):
            l1 = lists[l]
            l2 = lists[l + 1] if l + 1 < len(lists) else None
            merged = mergeList(l1, l2)
            mergedList.append(merged)
        lists = mergedList
    return lists[0]
    
def characterReplacement(s: str, k: int) -> int:
    l = 0
    maxLen = 0
    currMax = 0
    dic = {}
    for r in range(len(s)):
        dic[s[r]] = 1 + dic.get(s[r], 0)
        currMax = max(currMax, dic[s[r]])
        if (r - l + 1) - currMax > k:
            dic[s[l]] -= 1
            l += 1
        maxLen = max(maxLen, (r - l + 1))
    return maxLen


def validSudoku17(board: list[list[int]]) -> bool:
    rows = defaultdict(list)
    cols = defaultdict(list)
    box = defaultdict(list)
    for r in range(9):
        for c in range(9):
            if board[r][c] != '.':
                currChar = board[r][c]
                if currChar in rows[r] or currChar in cols[c] or currChar in box[(r//3, c//3)]:
                    return False
                rows[r].append(currChar)
                cols[c].append(currChar)
                box[(r//3, c//3)].append(currChar)
    return True


def  minWinSub(s: str, t: str) -> str:
    window = {}
    countT = {}
    for i in t:
        countT[i] = countT.get(i, 0) + 1
    l = 0
    have = 0
    need = len(countT)
    res = [-1, -1]
    resLen = float("inf")
    for r in range(len(s)):
        window[s[r]] = 1 + window.get(s[r], 0)
        if s[r] in countT and window[s[r]] == countT[s[r]]:
            have += 1
        while have == need:
            if resLen > (r - l + 1):
                resLen = r - l + 1
                res = [l, r]
            window[s[l]] -= 1
            if s[l] in countT and window[s[l]] < countT[s[l]]:
                have -= 1
            l += 1
    if res == float("inf"):
        return ""
    l, r = res
    return s[l: r + 1]
            

def reverseLinkList(head: Node) -> Node:
    if not head or not head.next:
        return head
    newHead = head
    head = reverseLinkList(head.next)
    newHead.next.next = newHead
    head.next = None
    return newHead
    

def containMostWater(heights: list[int]) -> int:
    maxWater = 0
    l = 0
    r = len(heights) - 1
    while l < r:
        water = (r - l) * min(heights[l], heights[r])
        maxWater = max(maxWater, water)
        if heights[l] <= heights[r]:
            l += 1
        else:
            r -= 1
    return maxWater
            

def reorderLinkedList(head: Node) -> Node:
    slow = head
    fast = head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next
        if fast.next:
            fast = fast.next
    half = slow.next
    slow.next = None
    prev = None
    while half:
        nxtNode = half.next
        half.next = prev
        prev = half
        half = nxtNode
    l1 = head
    l2 = prev
    
    while l1 and l2:
        temp1 = l1.next if l1.next else None
        temp2 = l2.next if l2.next else None
        l1.next = l2
        l2.next = temp1
        l1 = temp1
        l2 = temp2
    return head
        

def trappingRainWater(heights: list[int]) -> int:
    l = 0
    r = len(heights) - 1
    maxL = heights[l]
    maxR = heights[r]
    ans = 0
    while l <= r:
        if maxL <= maxR:
            water = min(maxL, maxR) - heights[l]
            if water > 0:
                ans += water
            maxL = max(maxL, heights[l])
            l += 1
        else:
            water = min(maxL, maxR) - heights[r]
            if water > 0:
                ans += water
            maxR = max(maxR, heights[r])
            r -= 1
    return ans


def permutationOfString(s1: str, s2: str) -> bool:
    if len(s1) != len(s2):
        return False
    hash1 = {}
    hash2 = {}
    for i in range(len(s1)):
        hash1[s1[i]] = 1 + hash1.get(s1[i], 0)
        hash2[s2[i]] = 1 + hash2.get(s2[i], 0)
    for i in hash1:
        if hash1[i] != hash2.get(i, 0):
            return False
    return True


def permuT(s1: str, s2: str) -> bool:
    countS1 = [0] * 26
    countS2 = [0] * 26
    if len(s1) > len(s2):
        return False
    for i in range(len(s1)):
        countS1[ord(s1[i]) - ord("a")] += 1
        countS2[ord(s2[i]) - ord("a")] += 1
    match = 0
    for i in range(26):
        if countS1[i] == countS2[i]:
            match += 1
    l = 0
    print(countS1)
    for r in range(len(s1), len(s2)):
        print(match)
        print(countS2)
        if match == 26:
            return True
        index = ord(s2[r]) - ord("a")
        countS2[index] += 1
        if countS2[index] == countS1[index]:
            match += 1
        elif countS2[index] - 1 == countS1[index]:
            match -= 1
        index = ord(s2[l]) - ord("a")
        countS2[index] -= 1
        if countS2[index] == countS1[index]:
            match += 1
        elif countS2[index] + 1 == countS1[index]:
            match -= 1
        l += 1
    
    if match == 26:
        return True
    return False
        

def bestTimeToBuyStock(prices: list[int]) -> int:
    l = 0
    maxProfit = 0
    for r in range(len(prices)):
        profit = prices[r] - prices[l]
        if profit >= 0:
            maxProfit = max(maxProfit, profit)
        else:
            l = r
    return maxProfit

def findDuplicateNumber(nums: list[int]) -> int:
    slow = 0
    fast = 0
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        print(slow, fast)
        if fast == slow:
            break
    newSlow = 0
    while True:
        slow = nums[slow]
        newSlow = nums[newSlow]
        if slow == newSlow:
            return slow
    

def longestConsecutive(nums: list[int]) -> int:
    st = set(nums)
    maxCnt = 0
    for i in range(len(nums)):
        curr = nums[i]
        cnt = 1
        while curr - 1 in st:
            curr -= 1
        while curr + 1 in st:
            curr += 1
            cnt += 1
        maxCnt = max(maxCnt, cnt)
    return maxCnt


def reverseLL(head: Node) -> Node:
    if not head or not head.next:
        return head
    
    newHead = reverseLL(head.next)
    head.next.next = head
    head.next = None
    return newHead


def slidingWinMax(nums: list[int], k: int) -> list[int]:
    ans = []
    l = 0
    q = deque()
    for r in range(len(nums)):
        while q and nums[q[-1]] < nums[r]:
            q.pop()
        q.append(r)
        while l > q[0]:
            q.popleft()
        if (r - l + 1) == k:
            ans.append(nums[q[0]])
            l += 1
    return ans


def threeSum(nums : list[int]) -> list[list[int]]:
    ans = []
    nums.sort()
    n = len(nums)
    for i, a in enumerate(nums):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        l = i + 1
        r = n - 1
        while l < r:
            target = a + nums[l] + nums[r]
            if target > 0:
                r -= 1
            elif target < 0:
                l += 1
            else:
                ans.append([a, nums[l], nums[r]])
                l += 1
                while nums[l] == nums[l - 1]:
                    l += 1
    return ans


def enodeStr(strs: list[str]) -> str:
    s = ""
    for i in strs:
        length = len(i)
        s += str(length) + "#" + i
    return s

def decodeStr(s: str) -> list[str]:
    i = 0
    ans = []
    while i < len(s):
        j = i
        while s[j] != "#":
            j += 1
        lenS = int(s[i:j])
        ans.append(s[j + 1: j + 1 + lenS])
        i = j + 1 + lenS
    return ans


def longestSubstringWithoutRepeatingCharacter10(s: str) -> int:
    l = 0
    maxLen = 0
    st = set()
    for r in range(len(s)):
        while st and s[r] in st:
            st.remove(s[l])
            l += 1
        st.add(s[r])
        maxLen = max(maxLen, (r - l + 1))
    return maxLen


def diameterOfBinaryTree(root) -> int:
    res = 0
    
    def dfs(root):
        if not root:
            return 0
        left = dfs(root.left)
        right = dfs(root.right)
        res = max(res, left + right)
        
        return 1 + max(left, right)
    dfs(root)
    return res
        


def prodExceptItself(nums: list[int]) -> list[int]:
    n = len(nums)
    prefix = [0] * n
    prod = 1
    for i in range(n):
        prod *= nums[i]
        prefix[i] = prod
    prod = 1
    postfix = [0] * n
    for i in range(n - 1, -1, -1):
        prod *= nums[i]
        postfix[i] = prod
    ans = [0] * n
    for i in range(n):
        if i == 0:
            ans[i] = 1 * postfix[i + 1]
            continue
        if i == n - 1:
            ans[i] = prefix[i - 1] * 1
            continue
        ans[i] = postfix[i + 1] * prefix[i - 1]
    return ans


def anagram(s: str, t: str) -> bool:
    hash1 = {}
    hash2 = {}
    length1 = len(s)
    length2 = len(t)
    if length1 != length2:
        return False
    for i in range(length1):
        hash1[s[i]] = 1 + hash1.get(s[i], 0)
        hash2[t[i]] = 1 + hash2.get(t[i], 0)
    for i in hash1:
        if hash1[i] != hash2.get(i, 0):
            return False
    return True


def largestRectangleInHisto20(heights: list[int]) -> int:
    st = [] #[ind, val]
    maxArea = 0
    for i in range(len(heights)):
        flag = False
        while st and st[-1][1] >= heights[i]:
            flag = True
            popped = st.pop()
            area = popped[1] * (i - popped[0])
            maxArea = max(maxArea, area)
        val = [i, heights[i]]
        if flag:
            val = [popped[0], heights[i]]
        st.append(val)
    while st:
        popped = st.pop()
        area = popped[1] * (len(heights) - popped[0])
        maxArea = max(area, maxArea)
    return maxArea

def medianOfTwoSortedArray21(nums1: list[int], nums2: list[int]) -> int:
    if len(nums1) < len(nums2):
        nums1, nums2 = nums2, nums1
    n = len(nums1)
    m = len(nums2)
    total = n + m
    half = total // 2
    l = 0
    r = m - 1
    while True:
        num2Mid = (l + r) // 2
        num1Mid = half - num2Mid - 2
        
        num2Left = nums2[num2Mid] if num2Mid >= 0 else float("-inf")
        num1Left = nums1[num1Mid] if num1Mid >= 0 else float("-inf") 
        num2Right = nums2[num2Mid + 1] if num2Mid + 1 < m else float("inf")
        num1Right = nums1[num1Mid + 1] if num1Mid + 1 < n else float("inf")
        
        if num2Left <= num1Right and num1Left <= num2Right:
            # This is the correct range
            if total % 2 == 0:
                return (max(num2Left, num1Left) + min(num1Right, num2Right)) / 2
            else:
                return min(num1Right, num2Right)
        elif num2Left > num1Right:
            r = num2Mid - 1
        else:
            l = num2Mid + 1


def searchRoatedArray(nums: list[int], target: int) -> int:
    l = 0
    h = len(nums) - 1
    while l <= h:
        m = (l + h) // 2
        if nums[m] == target:
            return m
        if nums[l] <= nums[m]:
            if nums[l] <= target and target <= nums[m]:
                h = m - 1
            else:
                l = m + 1
        else:
            if nums[h] >= target and target >= nums[m]:
                l = m + 1
            else:
                h = m - 1
    return -1

def merge2LinkedList(list1: Node, list2: Node) -> Node:
    dummy = Node(0)
    tail = dummy
    while list1 and list2:
        if list1.val <= list2.val:
            tail.next = list1
            list1 = list1.next
        else:
            tail.next = list2
            list2 = list2.next
        tail = tail.next
    if list1:
        tail.next = list1
    if list2:
        tail.next = list2
    return dummy.next


def evalReversePolish(tokens: list[str]) -> int:
    st = []
    for i in tokens :
        if i.isdigit() or (len(i) > 1 and i[1].isdigit()):
            st.append(int(i))
        else:
            operand1 = st.pop()
            operand2 = st.pop()
            ans = 0
            if i == "+":
                ans = operand2 + operand1
            elif i == "-":
                ans = operand2 - operand1
            elif i == "*":
                ans = operand2 * operand1
            elif i == "/":
                ans = int(operand2 / operand1)
            st.append(ans)
    return st[0]


def searchIn2DMatrix(matrix: list[list[int]], target: int) -> int:
    outL = 0
    outH = len(matrix) - 1
    while outL <= outH:
        outM = (outL + outH) // 2
        if target >= matrix[outM][0] and target <= matrix[outM][-1]:
            nums = matrix[outM]
            l = 0
            h = len(nums) - 1
            while l <= h:
                m = (l + h) // 2
                if nums[m] == target:
                    return m
                elif nums[m] >= target:
                    h = m - 1
                else:
                    l = m + 1
            return -1
        elif target >= matrix[outM][0] and target >= matrix[outM][-1]:
            outL = outM + 1
        else:
            outH = outM - 1
    return -1


def validParathensis(s: str) -> bool:
    st = []
    for i in s:
        if i in ['(', '[', '{']:
            st.append(i)
        else:
            if not st:
                return False
            popped = st.pop()
            if (i == ")" and popped != "(") or (i == "}" and popped != "{") or (i == "]" and popped != "["):
                return False
    if st:
        return False
    return True


def twoSum(nums: list[int], target: int) -> list[int]:
    hash = {}
    for i in range(len(nums)):
        if target - nums[i] in hash:
            return [hash[target - nums[i]], i]
        hash[nums[i]] = i
    return [-1, -1]
    

def containsDuplicates(nums: list[int]) -> bool:
    st = set()
    for i in range(len(nums)):
        if nums[i] in st:
            return True
        st.add(nums[i])
    return False


def binaryTreeLevelOrderTraversal(root):
    q = deque()
    q.append(root)
    ans = []
    while q:
        temp = []
        for i in range(len(q)):
            popped = q.popleft()
            if popped:
                temp.append(popped.val)
                q.append(popped.left)
                q.append(popped.right)
            
        if temp:
            ans.append(temp)
    return ans


def goodNodes(root):
    def dfs(root, maxVal):
        if not root:
            return 0
        res = 1 if root.val >= maxVal else 0
        maxVal = max(maxVal, root.val)
        res += dfs(root.left, maxVal)
        res += dfs(root.right, maxVal)
    return dfs(root, float("-inf"))

def copyLLWithRandomPointer(head: Node):
    dic = {None : None}
    it = head
    while it:
        dic[it] = Node(it.val)
        it = it.next
    it = head
    while it:
        dic[it].next = dic[it.next]
        dic[it].random = dic[it.random]
        it = it.next
    return dic[head]


def binaryTree(p, q):
    if not p and not q:
        return True
    if not p or not q:
        return False
    if p and q and p.val == q.val:
        return binaryTree(p.left, q.left) and binaryTree(p.right, q.right)
    return False

def sameBinaryTree(p, q):
    def binaryTree(p, q):
        if not p and not q:
            return True
        if p and q and p.val == q.val:
            return binaryTree(p.left, q.left) and binaryTree(p.right, q.right)
        return False
    
    
    if not q:
        return True
    if not p:
        return False
    if binaryTree(p, q):
        return True
    return sameBinaryTree(p.left, q) or sameBinaryTree(p.right, q)


def findDuplicateNumber(nums: list[int]) -> int:
    slow = 0
    fast = 0
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    newSlow = 0
    while True:
        slow = nums[slow]
        newSlow = nums[newSlow]
        if slow == newSlow:
            return slow
    

def sumOfLL(l1, l2):
    dummy = Node()
    tail = dummy
    carry = 0
    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        sum = val1 + val2 + carry
        carry = sum // 10
        sum = sum % 10
        tail.next = Node(sum)
        tail = tail.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    return dummy.next

def linkedListCycle(head):
    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next
        if fast:
            fast = fast.next
        if slow == fast:
            return True
    return False

def invertBinaryTree(root):
    if not root:
        return root
    root.left, root.right = root.right, root.left
    invertBinaryTree(root.left)
    invertBinaryTree(root.right)
    return root


def heightBalanced(root):
    ans = True
    def dfs(root):
        if not root:
            return 0
        left= root.left
        right = root.right
        
        if left - right > 1 or left - right < -1:
            ans = False
        return 1 + max(left, right)
    dfs(root)
    return ans
    

def permutationInString(s1, s2):
    dic1 = {}
    dic2 = {}
    for i in s1:
        dic1[i] = 1 + dic1.get(i, 0)
    print(dic1)
    l = 0
    ans = False
    for r in range(len(s2)):
        dic2[s2[r]] = 1 + dic2.get(s2[r], 0)
        if r - l + 1 == len(s1):
            flag = True
            for i in dic1:
                if dic1[i] != dic2.get(i, 0):
                    flag = False
                    break
            if flag:
                ans = True
                break
            dic2[s2[l]] -= 1
            l += 1
    return ans


def permutaionInStringMatch(s1, s2):
    hash1 = [0] * 26
    hash2 = [0] * 26
    for i in range(len(s1)):
        ind1 = ord(s1[i]) - ord("a")
        ind2 = ord(s2[i]) - ord("a")
        hash1[ind1] += 1
        hash2[ind2] += 1
    print(hash1)
    print(hash2)
    match = 0
    for i in range(26):
        if hash1[i] == hash2[i]:
            match += 1
    print(match)
    l = 0
    for r in range(len(s1), len(s2)):
        if match == 26:
            return True
        newChar = ord(s2[r]) - ord("a")
        hash2[newChar] += 1
        if hash2[newChar] == hash1[newChar]:
            match += 1
        elif hash2[newChar] == hash1[newChar] + 1:
            match -= 1

        oldChar = ord(s2[l]) - ord("a")
        hash2[oldChar] -= 1
        if hash2[oldChar] == hash1[oldChar]:
            match += 1
        elif hash2[oldChar] + 1 == hash1[oldChar]:
            match -= 1
        l += 1
        print(match, l, r)
    if match == 26:
        return True
    return False

def removeFromEndOfLL(head, n):
    fast = head
    while n > 0:
        fast = fast.next
        n -= 1
    slow = head
    while fast.next:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return head

    
def dailyTemp29(temperatures: list[int]) : 
    ans = [0] * len(temperatures)
    st = [] #[ind, temp]
    for i in range(len(temperatures)):
        if not st:
            st.append([i, temperatures[i]])
            continue
        while st and st[-1][1] < temperatures[i]:
            popped = st.pop()
            ans[popped[0]] = i - popped[0]
        st.append([i, temperatures[i]])
    return ans


def carFleet(position: list[int], speed: list[int], target: int) -> int:
    st = []
    
    combined = [[p, s] for p, s in zip(position, speed)]
    for i in sorted(combined, reverse=True):
        dist = target - i[0]
        time = dist/ i[1]
        if not st:
            st.append(time)
        if st and st[-1] < time:
            st.append(time)
    return len(st)
        
    
def twoSum2(nums: list[int], target: int) -> list[int]:
    l = 0
    r = len(nums) - 1
    while l < r:
        if nums[l] + nums[r] > target:
            r -= 1
        elif nums[l] + nums[r] < target:
            l += 1
        else:
            return [l + 1, r + 1]
    return [-1,-1]


def maxDepthOfBinaryTree(root):
    if not root:
        return 0
    return 1 + max(maxDepthOfBinaryTree(root.left), maxDepthOfBinaryTree(root.right))
    
def largestRectangle29(heights: list[int]) -> int:
    st = [] #[ind, ht]
    maxHt = 0
    for i in range(len(heights)):
        if not st or st[-1][1] < heights[i]:
            st.append([i, heights[i]])
            continue
        if st[-1][1] >= heights[i]:
            popped = st[-1]
            while st and st[-1][1] >= heights[i]:
                popped = st.pop()
                area = popped[1] * (i - popped[0])
                maxHt = max(maxHt, area)
            st.append([popped[0], heights[i]])
    print(st)
    for i in range(len(st)):
        popped = st.pop()
        area = popped[1] * (len(heights) - popped[0])
        print(area)
        maxHt = max(maxHt, area)
    return maxHt


def topK(nums: list[int], k: int):
    ans = [[] for i in range(len(nums) + 1)]
    cnt = {}
    for i in nums:
        cnt[i] = 1 + cnt.get(i, 0)
    print(cnt)
    for i in cnt:
        ans[cnt[i]].append(i)
    out = []
    print(ans)
    for i in range(len(nums), 0, -1):
        for j in ans[i]:
            out.append(j)
            k -= 1
            if k == 0:
                break
        if k == 0:
            break
    return out
        




def kokoEatingBanana(piles: list[int], h: int) -> int:
    def canKokoeat(piles: list[int], h: int, rate: int) -> bool:
        ans = 0
        for banana in piles:
            ans += math.ceil(banana / rate)
        if ans <= h:
            return True
        return False
    low = 1
    high = max(piles)
    ans = h
    while low <= high:
        rate = (low + high) // 2
        if canKokoeat(piles, h, rate):
            ans = rate
            high = rate - 1
        else:
            low = rate + 1
    return ans


def trappingRainWater(height: list[int]) -> int:
    l = 0
    r = len(height) - 1
    maxL = height[l]
    maxR = height[r]
    trappedRain = 0
    while l <= r:
        if maxL <= maxR:
            rain = maxL - height[l]
            if rain > 0:
                trappedRain += rain
            maxL = max(maxL, height[l])
            l += 1
        else:
            rain = maxR - height[r]
            if rain > 0:
                trappedRain += rain
            maxR = max(maxR, height[r])
            r -= 1
    return trappedRain


def minInRotatedArr(nums: list[int]):
    l = 0
    h = len(nums) - 1
    ans = float("inf")
    while l <= h:
        mid = (l + h) // 2
        if nums[l] <= nums[mid]:
            ans = min(ans, nums[l])
            l = mid + 1
        else:
            ans = min(ans, nums[mid])
            h = mid
    return ans
            

def binaryTreeLevelOrderTraversal(root):
    q = deque()
    q.append(root)
    ans = []
    while q:
        temp = []
        for i in range(len(q)):
            if q[i]:
                temp.append(q[i])
                q.append(q[i].left)
                q.append(q[i].right)
        if temp:
            ans.append(temp)
    return ans


def subsets(nums: list[int]) -> list[list[int]]:
    ans = []
    temp = []
    def dfs(i):
        if i == len(nums):
            ans.append(temp[:])
            return
        temp.append(nums[i])
        dfs(i + 1)
        temp.pop()
        dfs(i + 1)
    dfs(0)
    return ans
        
def isAlpha30(c: chr):
    if ((c >= "a" and c <= "z") or (c >= "A" and c <= "Z") or (c >= "0" and c <= "9")):
        return True
    return False

def validPalindrome30(s: str):
    l = 0
    h = len(s) - 1
    while l <= h:
        while not isAlpha30(s[l]):
            l += 1
        while not isAlpha30(s[h]):
            h -= 1
        if s[l].lower() != s[h].lower():
            return False
        l += 1
        h -= 1
    return True


def minimumWindowSubstring30(s: str, t: str):
    countS = {}
    countT = {}
    for i in t:
        countT[i] = 1 + countT.get(i, 0)
    have = 0
    need = len(countT)
    l = 0
    ans = [-1, -1]
    ansLen = float("inf")
    for r in range(len(s)):
        countS[s[r]] = 1 + countS.get(s[r], 0)
        if s[r] in countT and countS[s[r]] == countT[s[r]]:
            have += 1
        while have == need:
            if ansLen > (r - l + 1):
                ansLen = r - l + 1
                ans = [l, r]
            countS[s[l]] -= 1
            if s[l] in countT and countT[s[l]] - 1 == countS[s[l]]:
                have -= 1
            l += 1
    if ansLen == float("inf"):
        return ""
    left, right = ans
    return s[left: right + 1]



def generateParenthesis30(n: int):
    ans = []
    temp = []
    def recur(open: int, close: int):
        if open == close == n:
            ans.append("".join(temp))
            return
        if open < n:
            temp.append("(")
            recur(open + 1, close)
            temp.pop()
        if close < open:
            temp.append(")")
            recur(open, close + 1)
            temp.pop()
    recur(0, 0)
    return ans


    
def validPalindrome(board: list[list[chr]]) -> bool:
    rows = defaultdict(set)
    cols = defaultdict(set)
    box = defaultdict(set)
    for r in range(9):
        for c in range(9):
            currChar = board[r][c]
            if board != ".":
                if currChar in rows[r] or currChar in cols[c] or currChar in box[(r//3, c//3)]:
                    return False
                rows[r].add(currChar)
                cols[c].add(currChar)
                box[(r//3, c//3)].add(currChar)
    return True


def binaryTreeRight(root):
    q = deque()
    q.append(root)
    ans = []
    while q:
        temp = []
        for i in range(len(q)):
            popped = q.popleft()
            if popped:
                temp.append(popped.val)
                q.append(popped.left)
                q.append(popped.right)
        if temp:
            ans.append(temp[-1])
    return ans


def reverseLinked30(root):
    if not root or not root.next:
        return root
    newHead = reverseLinked30(root.next)
    root.next.next = root.next
    root.next = None
    return newHead



def lowestAnestor(root, p, q):
    if not root:
        return
    if root.val < p.val and root.val < q.val:
        return lowestAnestor(root.right)
    elif root.val > p.val and root.val > q.val:
        return lowestAnestor(root.left)
    else:
        return root


def maxDepth(root):
    if not root:
        return 0
    return 1 + (maxDepth(root.left), maxDepth(root.right))


def characterReplacement(s, k):
    l = 0
    maxLen = 0
    currMax = 0
    dic = {}
    for r in range(len(s)):
        dic[s[r]] = 1 + dic.get(s[r], 0)
        currMax = max(currMax, dic[s[r]])
        if (r - l + 1) - currMax > k:
            dic[s[l]] -= 1
            l += 1
        maxLen = max(maxLen, (r - l + 1))
    return maxLen


def countGoodNodes(root):
    ans = 0
    def dfs(root, currMax):
        if not root:
            return
        if root.val >= currMax:
            ans += 1
        currMax = max(currMax,root.val)
        dfs(root.left, currMax)
        dfs(root.right, currMax)
    dfs(root, root.val)
    return ans

def reverseLL333(root):
    if not root or not root.next:
        return root
    newHead = reverseLL333(root.next)
    root.next.next = root
    root.next = None
    return newHead


def generatePara333(n: int) -> list[list[str]]:
    ans = []
    temp = []
    def recur(open, close):
        if open == close == n:
            ans.append("".join(temp))
            return
        if open < n:
            temp.append("(")
            recur(open + 1, close)
            temp.pop()
        if close < open:
            temp.append(")")
            recur(open, close + 1)
            temp.pop()
    recur(0, 0)
    return ans


# def validBst(root):
#     self.ans = True
#     def dfs(root, leftVal, rightVal):
#         if self.ans == False:
#             return
#         if not root:
#             return
#         if root.val >= rightVal or root.val <= leftVal:
#             self.ans = False
#         dfs(root.left, leftVal, root.val)
#         dfs(root.right, root.val, rightVal)
#     dfs(root, float("-inf"), float("inf"))
#     return self.ans


def medianOfTwoSortedArray333(nums1: list[int], nums2: list[int]) -> int:
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    a, b = nums1, nums2
    n, m = len(a), len(b)
    total = n + m
    half = total // 2
    
    al = 0
    ah = n - 1
    while True:
        am = (al + ah) // 2
        bm = half - am - 2
        aleft = a[am] if am >= 0 else float("-inf")
        bleft = b[bm] if bm >= 0 else float("-inf")
        aright = a[am + 1] if am + 1 < n else float("inf")
        bright = b[bm + 1] if bm + 1 < m else float("inf")
        if aleft <= bright and aright >= bleft:
            if total % 2 == 0:
                return ((max(aleft, bleft) + min(aright, bright)) / 2)
            else:
                return (min(aright, bright))
        elif aleft > bright:
            ah = am - 1
        else:
            al = am + 1
    

def subsets(nums: list[int]):
    ans = []
    temp = []
    def recur(i: int, n: int = len(nums)):
        if i == n:
            ans.append(temp[:])
            return
        temp.append(nums[i])
        recur(i + 1, n)
        temp.pop()
        recur(i + 1, n)
    recur(0)
    return ans


def findDuplicateNumeber(nums: list[int]):
    slow = 0
    fast = 0
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    newSlow = 0
    while True:
        slow = nums[slow]
        newSlow = nums[newSlow]
        if slow == newSlow:
            return slow
    

def bestTimeToStock(prices: list[int]):
    maxPrice = 0
    l = 0
    for r in range(len(prices)):
        profit = prices[r] - prices[l]
        if profit >= 0:
            maxPrice = max(maxPrice, profit)
        else:
            l = r
    return maxPrice
            

def kthSmallest(root, k):
    cnt = [0]
    res = [None]
    
    def dfs(node):
        if not node or cnt[0] >= k:
            return
        dfs(node.left)
        if cnt[0] >= k:
            return
        cnt[0] += 1
        if cnt[0] == k:
            res[0] = node.val
            return
        dfs(node.right)
    dfs(root)
    return res[0]

    
def validSudoku(board):
    rows = defaultdict(set)
    cols = defaultdict(set)
    box = defaultdict(set)
    for r in range(9):
        for c in range(9):
            curr = board[r][c]
            if curr != ".":
                if curr in rows[r] or curr in cols[c] or curr in box[(r//3, c// 3)]:
                    return False
                rows[r].add(curr)
                cols[c].add(curr)
                box[(r//3, c// 3)].add(curr)
    return True


def binaryTreeMaxPathSum(root):
    ans = [0]
    
    def dfs(root):
        if not root:
            return 0
        left = dfs(root.left)
        right = dfs(root.right)
        left = max(left.val, 0)
        right = max(right.val, 0)
        ans[0] = max(ans[0], left + right + root.val)
        
        return root.val + max(left, right)
    return ans[0]


def diameterOfBinaryTree(root):
    ans = [0]
    def dfs(root):
        if not root:
            return 0
        left = dfs(root.left)
        right = dfs(root.right)
        ans[0] = max(ans[0], left + right)
        return max(left, right) + 1
    dfs(root)
    return ans[0]


def binaryLevelOrderTraversal(root):
    q = deque()
    q.append(root)
    ans = []
    while q:
        temp = []
        for i in range(len(q)):
            node = q.popleft()
            if node:
                temp.append(node.val)
                q.append(node.left)
                q.append(node.right)
            if temp:
                ans.append(temp)
    return temp

                

def largestRecInHis(heights: list[int]):
    st = []
    maxRec = 0
    n = len(heights)
    for i in range(len(heights)):
        if not st or st[-1][0] < heights[i]:
            st.append([heights[i], i])
        else:
            poppedInd = i
            while st and st[-1][0] >= heights[i]:
                popped = st.pop()
                poppedInd = popped[1]
                area = popped[0] * (i - poppedInd)
                maxRec = max(area , maxRec)
            st.append([heights[i], poppedInd])
    while st:
        popped = st.pop()
        area = popped[0] * (n - popped[1])
        maxRec = max(maxRec, area)
    return maxRec
                

def longestSubStringWithoutRepeatingCharacters(s: str) -> int:
    st = set()
    l = 0
    ans = 0
    for r in range(len(s)):
        while s[r] in st:
            st.remove(s[l])
            l += 1
        st.add(s[r])
        ans = max(ans, r - l + 1)
    return ans
                

def threeSum99(nums : list[int]):
    ans = []
    n = len(nums)
    nums.sort()
    for i, a in enumerate(nums):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        j = i + 1
        k = n - 1
        while j < k:
            target = a + nums[j] + nums[k]
            if target > 0:
                k -= 1
            elif target < 0:
                j += 1
            else:
                ans.append([a, nums[j], nums[k]])
                j += 1
                while j < k and nums[j] == nums[j - 1]:
                    j += 1
    return ans


def combinationSum(candidats: list[int], target: int):
    temp = []
    ans = []
    sum = 0
    n = len(candidats)
    def recur(sum, start):
        if sum > target or start >= n:
            return
        if sum == target:
            ans.append(temp[:])
            return
        temp.append(candidats[start])
        sum += candidats[start]
        recur(sum, start)
        temp.pop()
        sum -= candidates[start]
        recur(sum, start + 1)
    recur(0, 0)
    return ans

    
def combinationSum2(candidates: list[int], target: int) -> list[int]:
    candidates.sort()
    temp = []
    ans = []
    n = len(candidates)
    def recur(sum, start):
        if sum == target:
            ans.append(temp[:])
            return
        if sum > target or start == n:
            return
        
        temp.append(candidates[start])
        recur(sum + candidates[start], start + 1)
        temp.pop()
        while start + 1 < n and candidates[start] == candidates[start + 1]:
            start += 1
        recur(sum, start + 1)
    recur(0, 0)
    return ans
        
    
def binaryTreeMaxPathSum(root):
    ans = [0]
    def dfs(root):
        if not root:
            return 0
        left = dfs(root.left)
        right = dfs(root.right)
        left = max(left, 0)
        right = max(right, 0)
        ans[0] = max(ans[0], left + right + root.val)
        
        return root.val + max(left, right)
    dfs(root)
    return ans[0]



def balancedBinaryTree1(root):
    ans = [True]
    def dfs(root):
        if not root:
            return 0
        left = dfs(root.left)
        right = dfs(root.right)
        if ans[0] == False:
            return
        if left - right > 1 or left - right < -1:
            ans[0] = False
        return 1 + max(left, right)
    dfs(root)
    return ans[0]


def validParenthesis(s: str):
    st = []
    for i in s:
        if i == "(" or i == "[" or i == "{":
            st.append(i)
        else:
            if not st:
                return False
            popped = st.pop()
            if not(i == ")" and popped == "(" or i == "]" and popped == "[" or i == "}" and popped == "{"):
                return False
    if st:
        return False
    return True


def searchInRotatedSortedArray111(nums: list[int], target: int) -> int:
    l = 0
    h = len(nums) - 1
    while l <= h:
        m = (l + h) // 2
        if nums[m] == target:
            return m
        elif nums[l] <= nums[m]:
            if nums[l] <= target and target <= nums[m]:
                h = m - 1
            else:
                l = m + 1
        else:
            if nums[h] >= target and nums[m] <= target:
                l = m + 1
            else:
                h = m - 1
    return -1


def longestConsecutiveSequence11(nums: list[int]):
    st = set(nums)
    maxCnt = 0
    for i in range(len(st)):
        a = nums[i]
        while a - 1 in st:
            a -= 1
        cnt = 0
        while a + 1 in st:
            a += 1
            cnt += 1
        maxCnt = max(maxCnt, cnt + 1)
    return maxCnt
        
        
def merge2SortedLL(list1, list2):
    dummy = ListNode()
    tail = dummy
    while list1 and list2:
        if list1.val <= list2.val:
            tail.next = list1
            list1 = list1.next
        else:
            tail.next = list2
            list2 = list2.next
        tail = tail.next
    if list1:
        tail.next = list1
    if list2:
        tail.next = list2
    return dummy.next




def permutation11(nums: list[int]) -> list[list[int]]:
    if len(nums) == 0:
        return [[]]
    perms = permutation11(nums[1:])
    res = []
    for elem in perms:
        for i in range(len(elem) + 1):
            elem_copy = elem[:]
            elem_copy.insert(i, nums[0])
            res.append(elem_copy)
    return res



def wordSearch(board: list[list[int]], word: str):
    ROWS, COLS = len(board), len(board[0])
    path = set()
    def dfs(r, c, i):
        if i >= len(word):
            return True
        if r < 0 or c < 0 or r >= ROWS or c >= COLS or board[r][c] != word[i] or (r,c) in path:
            return False
        path.add((r,c))
        res = dfs(r + 1, c, i + 1) or dfs(r - 1, c, i + 1) or dfs(r, c + 1, i + 1) or dfs(r, c - 1, i + 1)
        path.remove((r,c))
        return res
    for r in range(ROWS):
        for c in range(COLS):
            if dfs(r,c, 0):
                return True
    return False


    
def palindromePartitioning(s: str):
    n = len(s)
    # Precompute palindrome table using dynamic programming
    dp = [[False] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            if i == j:
                dp[i][j] = True  # Single character is a palindrome
            elif s[i] == s[j]:
                if j == i + 1 or dp[i + 1][j - 1]:
                    dp[i][j] = True  # Two same characters or substring between is a palindrome

    result = []
    current_partition = []

    def dfs(start):
        if start >= n:
            result.append(current_partition.copy())
            return
        for end in range(start, n):
            if dp[start][end]:
                current_partition.append(s[start:end+1])
                dfs(end + 1)
                current_partition.pop()

    dfs(0)
    return result
             

def letterCombinations(digits: str):
    res = []
    digitToChar = { "2": "abc",
                        "3": "def",
                        "4": "ghi",
                        "5": "jkl",
                        "6": "mno",
                        "7": "pqrs",
                        "8": "tuv",
                        "9": "wxyz"
    }
    def dfs(i, curStr):
        if len(curStr) == len(digits):
            res.append(curStr)
            return
        for c in digitToChar[digits[i]]:
            dfs(i + 1, curStr + c)
    if digits:
        dfs(0, "")
    return res
    
def countPrimes(n):
    seive = [False] * n
    seive[0] = seive[1] = True
    for i in range(4,n,2):
        seive[i] = True

    for i in range(3, int(n ** 0.5) + 1, 2):
        if not seive[i]:
            for j in range(i * i, n, 2 * i):
                seive[j] = True
    return seive.count(False)


def prodOfArrExpectSelf(nums : list[int]):
    n = len(nums)
    pred = [1] * n
    succ = [1] * n
    temp = 1
    for i in range(n):
        temp *= nums[i]
        pred[i] = temp
    temp = 1
    for i in range(n-1, -1, -1):
        temp *= nums[i]
        succ[i] = temp
    ans = [1] * n
    for i in range(n):
        if i == 0:
            ans[0] = succ[1] * 1
            continue
        elif i == n - 1:
            ans[n - 1] = pred[n - 2] * 1
            continue
        ans[i] = succ[i + 1] * pred[i - 1]
    return ans


class NumArray():
    def __init__(self, nums):
        self.arr = [0] * (len(nums))
        temp = 0
        for i in range(len(nums)):
            temp += nums[i]
            self.arr[i] = temp
        
    def sumRange(self, left, right):
        leftSum = self.arr[left - 1] if left - 1 >= 0 else 0
        rightSum = self.arr[right]
        return rightSum - leftSum


class NumMatrix():

    def __init__(self, matrix):
        ROWS = len(matrix)
        COLS = len(matrix[0])
        self.sumMat = [[0] * (COLS+ 1) for r in range(ROWS + 1)]
        for r in range(ROWS):
            prefix = 0
            for c in range(COLS): 
                prefix += matrix[r][c]
                above = self.sumMat[r][c + 1]
                self.sumMat[r + 1][c + 1] = prefix + above
        

    def sumRegion(self, row1, col1, row2, col2):
        row1, col1, row2, col2 = row1 + 1, col1 + 1, row2 + 1, col2 + 1
        
        bottomRight = self.sumMat[row2][col2]
        above = self.sumMat[row1 - 1][col2]
        left = self.sumMat[row2][col1 - 1]
        topLeft = self.sumMat[row1 - 1][col1 - 1]
        
        return bottomRight - above - left + topLeft

matrix = [[3, 0, 1, 4, 2], [5, 6, 3, 2, 1], [1, 2, 0, 1, 5], [4, 1, 0, 1, 7], [1, 0, 3, 0, 5]]
narr = NumMatrix(matrix)
print(narr.sumRegion(1, 2, 2, 4))