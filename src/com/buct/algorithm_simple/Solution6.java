package com.buct.algorithm_simple;

import java.util.Arrays;

/**
 * 合并两个排序的整数数组A和B变成一个新的数组。
 */
public class Solution6 {
    /*
     * @param A: sorted integer array A
     * @param B: sorted integer array B
     * @return: A new sorted integer array
     */
    public static int[] mergeSortedArray(int[] A, int[] B) {
        // write your code here
        int[] C = new int[A.length + B.length];
        int index = 0;
        for (int i = 0; i < A.length; i++) {
            C[index++] = A[i];
        }
        for (int i = 0; i < B.length; i++) {
            C[index++] = B[i];
        }
        System.out.println(Arrays.toString(C));
        for (int i = A.length; i <= C.length-1; i++) {
            int temp = C[i];
            int j = i-1;
            for (; j >=0; j--) {
                if (C[j] >= temp) {
                    C[j + 1] = C[j];
                } else {
                    break;
                }
            }
            C[j+1] = temp;
        }
        return C;
    }

    public static void main(String[] args) {
        int[] c = new int[5];
        c[0] = 12;
        int[] a = {1,5};
        int[] b = {2,3};

        System.out.println(Arrays.toString(mergeSortedArray(a,b)));
    }
}