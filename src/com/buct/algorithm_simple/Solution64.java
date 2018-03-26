package com.buct.algorithm_simple;

/**
 * 合并两个排序的整数数组A和B变成一个新的数组。
 */
public class Solution64 {
    /*
     * @param A: sorted integer array A which has m elements, but size of A is m+n
     * @param m: An integer
     * @param B: sorted integer array B which has n elements
     * @param n: An integer
     * @return: nothing
     */
    public void mergeSortedArray(int[] A, int m, int[] B, int n) {
        // write your code here
        for (int i = 0; i < n; i++) {
            int temp = B[i];
            int j = m-1+i;
            for (; j >=0 ; j--) {
                if (A[j] > temp) {
                    A[j + 1] = A[j];
                } else {
                    break;
                }
            }
            A[j+1] = temp;
        }

    }
}