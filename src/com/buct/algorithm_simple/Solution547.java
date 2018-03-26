package com.buct.algorithm_simple;

import java.util.HashSet;

/**
 * 返回两个数组的交
 */
public class Solution547 {

    /*
     * @param nums1: an integer array
     * @param nums2: an integer array
     * @return: an integer array
     */
    public int[] intersection(int[] nums1, int[] nums2) {
        // write your code here
        //使用hashset  无序且唯一
        HashSet<Integer> hs1 = new HashSet<>();
        HashSet<Integer> hs2 = new HashSet<>();
        for (Integer i : nums1
                ) {
            hs1.add(i);
        }
        for (Integer i : nums2
                ) {
            if (hs1.contains(i)) {
                hs2.add(i);
            }
        }
        int[] section = new int[hs2.size()];
        int index = 0;
        for (Integer i: hs2
             ) {
            section[index++] = i;
        }
        return section;
    }
};