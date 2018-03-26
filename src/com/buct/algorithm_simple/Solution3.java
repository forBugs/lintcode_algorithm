package com.buct.algorithm_simple;

/**
 * 计算数字k在0到n中的出现的次数，k可能是0~9的一个值
 */
public class Solution3 {
    /*
     * @param : An integer
     * @param : An integer
     * @return: An integer denote the count of digit k in 1..n
     */
    public int digitCounts(int k, int n) {
        // write your code here
        int count = 0;
        for (int i = 0; i <=n; i++) {
            char k_char = String.valueOf(k).charAt(0);
            String i_str = String.valueOf(i);
            for (int j = 0; j < i_str.length(); j++) {
                if (i_str.charAt(j) ==k_char ) {
                    count++;
                }
            }
        }
        return count;
        
    }
};