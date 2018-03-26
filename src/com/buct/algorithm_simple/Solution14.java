package com.buct.algorithm_simple;

import java.util.HashSet;

/**
 * 二分查找：给定一个排序的整数数组（升序）和一个要查找的整数target，用O(logn)的时间查找到target第一次出现的下标（从0开始），如果target不存在于数组中，返回-1。
 */
class Solution14 {
    /**
     * @param nums: The integer array.
     * @param target: Target to find.
     * @return: The first position of target. Position starts from 0.
     */
    public int binarySearch(int[] nums, int target) {
        //write your code here


        int low = 0;
        int high = nums.length - 1;
        int middle = 0;
        while (low + 1 < high) {
            middle = low + (high-low)/2;
            if (nums[middle] < target) {
                low = middle;
            } else {
                high = middle;
            }
        }
        if (nums[low] == target) {
            return low;
        }
        if (nums[high] == target) {
            return high;
        }
        return -1;
    }
}