package com.datastructure.sort;

import org.junit.Test;

import java.util.Arrays;

/**
 * 快速排序  时间复杂度  n*logn   不稳定
 * Created by Administrator on 2018/3/21.
 */
public class QuickSort {
    //
    @Test
    public void test() {
        int a[] = {3,1,5,7,2,4,9,6,10,8};
        quickSort(a, 0, a.length - 1);
        System.out.println(Arrays.toString(a));
    }

    public void quickSort(int[] arr, int low, int high) {
        if (low < high) {  // 如果缺少条件 会发生堆栈溢出
            int middle = getMiddle(arr, low, high);
            quickSort(arr, low, middle - 1);
            quickSort(arr, middle + 1, high);
        }
    }

    //
    private int getMiddle(int[] arr, int low, int high) {
        int temp = arr[low]; // 基准元素 即首元素
        while (low < high) {
            while (low < high && arr[high] > temp) {
                //从从high开始找比基准小的，与low换位置
                high--;
            }
            arr[low] = arr[high];

            while (low < high && arr[low] < temp) {
                // 从low开始找比基准大的  与high交换位置
                low++;
            }
            arr[high] = arr[low];
        }
        // 找到最终基准元素的位置即low
        arr[low] = temp;
        return low;
    }



}

