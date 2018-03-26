package com.buct.algorithm_simple;

import java.util.List;

/**
 * 给定一个整数数组，找到一个具有最小和的子数组。返回其最小和。
 */
public class Solution44 {
    /*
     * @param nums: a list of integers
     * @return: A integer indicate the sum of minimum subarray
     */
    public int minSubArray(List<Integer> nums) {
        // write your code here
        if (nums.size() == 0 || nums == null) {
            return 0;
        }
        int curMin = nums.get(0);
        int sumMin = nums.get(0);
        for(int i = 1; i<nums.size(); i++) {
            curMin = Math.min(nums.get(i), curMin+nums.get(i));
            sumMin = Math.min(sumMin,curMin);
        }

        return sumMin;
    }
}