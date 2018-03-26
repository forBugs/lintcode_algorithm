package com.buct.algorithm_simple;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * 给定一个旋转排序数组，在原地恢复其排序。
 * 这道题主要考察观察能力，显然旋转数组，不管怎么旋转，由于之前是排好序的，所以找到比第一个数小的第一个数一定是整个排序数组中最小的数。
 * 所以，只要找到最小的那个数，然后把前面的数全部按顺序加到后面，最后再将前面的数删除就可以了
 *
 *
 *
 */
public class Solution39 {
    /*
     * @param nums: An integer array
     * @return: nothing
     */
    public static void recoverRotatedSortedArray(List<Integer> nums) {
        // write your code here
        int temp = nums.get(0);
        int i;
        for(i=0; i<nums.size(); i++) {
            if (nums.get(i) < temp) {
                break;
            }
        }

        if (i < nums.size()) {
            for (int j = 0; j < i; j++) {
                nums.add(nums.get(j));
            }

            nums.subList(0,i).clear();
        }
    }

    public static void main(String[] args) {
        List<Integer> list = new ArrayList<>();
        list.add(4);
        list.add(5);
        list.add(1);
        list.add(2);
        list.add(3);
        recoverRotatedSortedArray(list);
    }
}