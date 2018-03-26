package com.datastructure.sort;

import org.junit.Test;

import java.util.Arrays;

/**
 * 冒泡排序
 * Created by Administrator on 2018/3/20.
 */
public class BubbleSort {

    @Test
    public void test() {
        int[] array = {39, 44, 1, 0, 8, 66, 23};
        System.out.println(Arrays.toString(bubbleSort3(array)));
    }

    public int[] bubbleSort(int[] arr) {
        // 外城循环控制排序趟数 n-1趟
        for (int i = 0; i < arr.length - 1; i++) {
            // 内层循环控制每一趟排序多少次
            for (int j = 0; j < arr.length-1-i; j++) {
                if (arr[j] > arr[j+1]){
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j+1] = temp;
                }
            }
        }
        return arr;
    }

    // 优化1  设置一个标志位flag,用来判断本轮排序有没有交换，如果没有发生交换 退出循环
    public int[] bubbleSort2(int[] arr) {
        // 外城循环控制排序趟数 n-1趟
        boolean flag = true;
        for (int i = 0; i < arr.length - 1; i++) {
            // 内层循环控制每一趟排序多少次
            for (int j = 0; j < arr.length-1-i; j++) {
                if (arr[j] > arr[j+1]){
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j+1] = temp;
                    flag = false;
                }
            }
            if (flag){
                //如果没有发生交换 退出循环
                break;
            }
        }
        return arr;
    }


    //优化二：优化内层循环
    public int[] bubbleSort3(int[] arr) {
        // 外城循环控制排序趟数 n-1趟
        for (int i = 0; i < arr.length - 1; i++) {
            //标记最后一次交换的位置
            int flag = 0;
            // 内层循环控制每一趟排序多少次
            for (int j = 0; j < arr.length-1-i; j++) {
                if (arr[j] > arr[j+1]){
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j+1] = temp;
                    //当位置发生变换  flag的值就发生变化
                    flag = 1;
                }
            }
            if (flag==0){
                //如果没有发生交换 退出循环
                break;
            }
        }
        return arr;
    }
}
