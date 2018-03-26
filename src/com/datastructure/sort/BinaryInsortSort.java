package com.datastructure.sort;

import org.junit.Test;

import java.util.Arrays;

/**
 * 二分法插入排序  与插入排序相比减少了时间复杂度
 * Created by Administrator on 2018/3/20.
 */
public class BinaryInsortSort {

    public int[] binaryInsortSort(int[] arr){
        for (int i = 1; i < arr.length; i++) {
            int temp = arr[i];  // 待插入的元素
            int left = 0;
            int right = i -1;
            int middle = 0;
            // 利用二分法找到插入位置
            while (left <= right) {
                middle = (left + right)/2;
                if(arr[middle]<=temp){
                    left = middle + 1;
                } else {
                    right = middle - 1;
                }
            }

            // 将left及其之后的元素后移  left即为插入位置
            for (int j = i-1; j >= left; j--) {
                arr[j+1] = arr[j];
            }

            // 插入位置赋值
            if(left != i) {
                arr[left] = temp;
            }
        }
        return arr;
    }

    @Test
    public void test() {
        int[] a = {10, 8, 11, 3, 4, 11, 18, 33, -1, 0, 7};
        System.out.println(Arrays.toString(binaryInsortSort(a)));
    }
}
