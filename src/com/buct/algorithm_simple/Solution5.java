package com.buct.algorithm_simple;

import org.junit.Test;

/**
 * 在数组中找到第k大的元素
 */
class Solution5 {
    /*
     * @param k : description of k
     * @param nums : array of nums
     * @return: description of return
     */
    public int kthLargestElement(int k, int[] nums) {
        // write your code here
        return quickSort(nums,0,nums.length-1,k);

    }

    public int quickSort(int[] nums,int left,int right,int k){
        int i = left;
        int j = right;
        int temp = nums[i];
        while (i<j){
            while(i<j && temp>=nums[j]) j--;
            if (i<j){
                nums[i] = nums[j];
                i++;
            }

            while (i<j && temp<nums[i]) i++;
            if(i<j) {
                nums[j] = nums[i];
                j--;
            }

        }
        if(i == k-1){
            return temp;
        } else if(i < k-1){
            return quickSort(nums,i+1,right,k);
        } else {
            return quickSort(nums,left,i-1,k);
        }
    }

    @Test
    public void test(){
    }
}