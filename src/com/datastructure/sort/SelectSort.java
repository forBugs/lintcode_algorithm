package com.datastructure.sort;

import org.junit.Test;

import java.util.Arrays;

/**
 * 简单选择排序
 * Created by Administrator on 2018/3/20.
 */
public class SelectSort {
    public int[] selectSort(int[] arr) {
        int min,minIndex;
        int temp;
        for (int i = 0; i < arr.length; i++) {
            min = arr[i];
            minIndex = i;
            for (int j = i+1; j <arr.length ; j++) {
                //找到本轮最小值  并与指定的元素交换
                if (arr[j]< min) {
                    minIndex = j;
                }
            }
            // 交换最小值  与 相应的位置
            temp = arr[minIndex];
            arr[minIndex] = arr[i];
            arr[i] = temp;
        }
        return arr;
    }

    public void swap(int a, int b) {
        int temp;
        temp = a;
        a = b;
        b = temp;
    }

    @Test
    public void test() {
        int[] a = {9,4,2,6,7,3,10,33,88,1,17};
        System.out.println(Arrays.toString(selectSort(a)));
    }
}
