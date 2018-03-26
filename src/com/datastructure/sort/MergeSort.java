package com.datastructure.sort;

import org.junit.Test;

import java.lang.reflect.WildcardType;
import java.util.Arrays;

/**
 * 归并排序
 * Created by Administrator on 2018/3/21.
 */
public class MergeSort {
    @Test
    public void test() {
        int [] a = new int[]{90,3,2,67,44,-9,87,65,11,9,2,8};
        mergeSort(a, 0, a.length - 1);
        System.out.println(Arrays.toString(a));
    }
    // 递归进行排序
    public void mergeSort(int[] arr, int left, int right) {
        if (left < right) {
            // 找出中间索引
            int middle = (left + right) / 2;
            // 对左边数组进行递归排序
            mergeSort(arr, left, middle);
            mergeSort(arr, middle + 1, right);
            // 进行数组合并
            merge(arr,left,middle,right);
        }
    }

    // 进行二路归并  将左边有序数组与右边有序数组合并
    private void merge(int[] arr, int left, int middle, int right) {
        // 临时数组
        int[] tmpArray = new int[arr.length];
        // 右数组的第一个元素索引
        int rightStart = middle + 1;  //
        // 临时数组的第一个元素索引
        int third = left;
        //缓存左数组第一个元素的索引
        int tmp = left;
        while (left<=middle && rightStart<=right) {
            // 同时遍历俩个数组 将值较小的放入临时数组
            if(arr[left] <= arr[rightStart]) {
                tmpArray[third++] = arr[left++];
            } else {
                tmpArray[third++] = arr[rightStart++];
            }
        }
        // 如果左边数组还有剩余数据 把左边数组剩下的拷贝到新数组
        while (left <= middle) {
            tmpArray[third++] = arr[left++];
        }
        // 如果右边数组还有剩余数据......
        while (rightStart <= right) {
            tmpArray[third++] = tmpArray[rightStart++];
        }

        // 将新数组中的元素赋值给arr
        while (tmp <= right) {
            arr[tmp] = tmpArray[tmp];
            tmp++;
        }

    }

}
