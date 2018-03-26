package com.buct.algorithm_basic;

import java.util.Arrays;

/**
 * Created by Administrator on 2017/11/25.
 */
public class Sort {
    public static void sortIntegers1(int[] A) {
        // write your code here
        System.out.println("原始数组：" + Arrays.toString(A));

        for (int i = 1; i < A.length; i++) {
            //插入排序
            int temp = A[i];
            int j = i - 1;
            for (; j >= 0; j--) {
                if (A[j] > temp) {
                    A[j + 1] = A[j];
                } else {
                    break;
                }

            }
            A[j + 1] = temp;


        }

        System.out.println("排序后数组：" + Arrays.toString(A));


    }

    /**
     * 选择排序
     * @param A
     */
    public static void sortIntegers2(int[] A) {
        System.out.println("排序前：" + Arrays.toString(A));
        for (int i = 0; i < A.length -1 ; i++) {
            int k = i;
            int min = A[i];
            for (int j = i; j < A.length; j++) {
                if (A[j] < min) {
                    min = A[j];
                    k = j;
                }
            }
            //交换A[i] yu A[j]
            int temp = A[i];
            A[i] = A[k];
            A[k] = temp;
        }

        System.out.println("排序后：" + Arrays.toString(A));

    }

    public static void main(String[] args) {
        int[] a = {2, 5, -2, 1, 8};
        sortIntegers2(a);
    }
}
