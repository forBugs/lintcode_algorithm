package com.buct.algorithm_simple;

import java.util.PrimitiveIterator;

/**
 * 设计一个算法，计算出n阶乘中尾部零的个数
 * 设f(n)为函数值：
 *          0<n<5  f(n!)=0
 *          n>=5  f(n!)=k + f(k!)  k=n/5
 */
public class Solution2 {
    /*
     * @param n: An integer
     * @return: An integer, denote the number of trailing zeros in n!
     */
    public static long trailingZeros(long n) {
        // write your code here, try to do it without arithmetic operators.
        if (n>0 && n<5) return 0;
        long k = n/5;
        return k+trailingZeros(k);

    }
    public static long trailingZeros2(long n) {
        // write your code here, try to do it without arithmetic operators.
        int sum = 0;
        if (n < 0) return -1;
        while ((n /= 5) > 0) {
            sum += n;

        }
        return sum;

    }

    /**
     * 阶乘
     * @param n
     * @return
     */
    private static long factorical(long n) {

        if (n==1) return 1;
        return factorical(n-1)*n;
    }

    public static void main(String[] args) {
        System.out.println(trailingZeros(11));
    }
}