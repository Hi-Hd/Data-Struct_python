from collections import defaultdict

def groupAnagrams(strs: list[str]) -> list[list[str]]:
    res = defaultdict(list) #mapping charCount to list of anagrams
    for s in strs:
        count = [0] * 26
        for c in s:
            count[ord(c) - ord("a")] += 1
        res[tuple(count)].append(s)
    return res.values()

strs = ["eat","tea","tan","ate","nat","bat"]
groupAnagrams(strs)
