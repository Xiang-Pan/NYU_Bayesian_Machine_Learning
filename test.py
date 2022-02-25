'''
Author: Xiang Pan
Date: 2021-10-19 19:45:46
LastEditTime: 2021-10-19 19:51:43
LastEditors: Xiang Pan
Description: 
FilePath: /NYU_Bayesian_Machine_Learning/test.py
xiangpan@nyu.edu
'''
class Solution:
    def minCost(self, costs: List[List[int]]) -> int:
        numHouses = len(costs)
        dp = [[0]*3 for _ in range(numHouses)]
        dp[0] = costs[0]
        for index in range(1, numHouses):
            minCostByColorPreviousHouse = dp[index-1]   # get the previous dp status
            for currentColor in range(3):               # loop over current color
                colorCost = costs[index][currentColor]  # get current color cost
                minCostForPreviousHouse = min(minCostByColorPreviousHouse[:currentColor] + minCostByColorPreviousHouse[currentColor+1:]) #cobination min
                dp[index][currentColor] =  minCostForPreviousHouse + currentColor   # final status for current color choice
        return min(dp[-1])