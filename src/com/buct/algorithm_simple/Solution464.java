package com.buct.algorithm_simple;

import java.util.Arrays;

/**
 * 给一组整数，按照升序排序。使用归并排序，快速排序，堆排序或者任何其他 O(n log n) 的排序算法。
 */
public class Solution464 {
    /*
     * @param A: an integer array
     * @return: 
     */
    public static void sortIntegers2(int[] A) {
        // write your code here
        //使用快速排序
        quickSort(A,0,A.length-1);
    }

    public static void main(String[] args) {
        int[] a = {3,22,158,4,5};
        sortIntegers2(a);
        System.out.println(Arrays.toString(a));
    }
    private static void quickSort(int[] a, int low, int height) {
        if (low > height) {
            return;
        }
        int index = a[low];
        int i = low, j = height;
        while (i < j) {
            while (i<j && a[j]>=index){
                j--;
            }
            if (i<j){
                a[i++] = a[j];
            }

            while (i < j && a[i] < index) {
                i++;
            }
            if (i < j) {
                a[j--] = a[i];
            }
        }
        //吧index给a[i]
        a[i] = index;
        //递归左边和右边
        quickSort(a, 0, i - 1);
        quickSort(a,i+1,height);
    }
}