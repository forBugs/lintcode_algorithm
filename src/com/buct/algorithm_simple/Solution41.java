package com.buct.algorithm_simple;

/**
 * 最大子数组
 * 给定一个整数数组，找到一个具有最大和的子数组，返回其最大和
 */
public class Solution41 {
    /**
     * @param nums: A list of integers
     * @return: A integer indicate the sum of max subarray
     */
    public int maxSubArray(int[] nums) {
        // write your code here
        if(nums==null || nums.length == 0) return 0;

        int maxSum = nums[0], sum = nums[0]<0?nums[0]:0;
        for (int num: nums) {
            if(num < 0) maxSum = Math.max(maxSum,sum);
            sum = Math.max(maxSum,sum+num);
        }
        maxSum = Math.max(maxSum,sum);
        return maxSum;
    }
}