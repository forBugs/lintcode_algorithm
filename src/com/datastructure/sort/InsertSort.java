package com.datastructure.sort;

import org.junit.Test;

import java.util.Arrays;

/**
 * Created by Administrator on 2018/3/20.
 */
public class InsertSort {

    /**
     * 直接插入排序
     * @param arr
     */
    public int[] insertSort(int[] arr){
        for (int i = 1; i < arr.length; i++) {
            int temp = arr[i];  //待插入的值
            int j;
            for (j = i-1; j >=0 ; j--) {
                if (arr[j] > temp) {
                    arr[j + 1] = arr[j];  //后移一位  知道找到第一个比temp小
                } else {
                    break;
                }
            }
            //找到插入位置
            arr[j + 1] = temp;

        }
        return arr;
    }

    @Test
    public void test() {
        int[] a = {3, 1, 5, 7, 2, 4, 9, 6, 10, 8};
        System.out.println(Arrays.toString(insertSort(a)));
    }
}
