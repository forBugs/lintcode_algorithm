package com.buct.algorithm_simple;

import org.junit.Test;

public class Solution31 {
    /**
     * @param nums: The integer array you should partition
     * @param k: An integer
     * @return: The index after partition
     */
    public int partitionArray(int[] nums, int k) {
        // write your code here
        if(nums==null && nums.length == 0) return 0;
        int start = 0,end = nums.length - 1;
        while (true){
            while (start < end && nums[start] < k) start++;
            while (start < end && nums[end] >= k) end--;
            if(start>=end) break;

            swap(nums,start,end);
        }
        return nums[end] < k?end+1 : end;

    }

    /**
     * swap the arrays
     * @param nums
     * @param x
     * @param y
     */
    private void swap(int[] nums,int x,int y){
        int tmp = nums[x];
        nums[x] = nums[y];
        nums[y] = tmp;
    }

    @Test
    public void test(){
        int[] a = {3,8,37,2,4};
        int i = partitionArray(a, 6);
        System.out.println(i);
    }
}