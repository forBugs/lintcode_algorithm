package com.buct.algorithm_simple;

import org.junit.Test;

import java.util.Arrays;

/**
 * 两数之和
 * 给一个整数数组，找到两个数使得他们的和等于一个给定的数 target。

 你需要实现的函数twoSum需要返回这两个数的下标, 并且第一个下标小于第二个下标。注意这里下标的范围是 0 到 n-1。
 */
public class Solution56 {
    /**
     * @param numbers: An array of Integer
     * @param target: target = numbers[index1] + numbers[index2]
     * @return: [index1 + 1, index2 + 1] (index1 < index2)
     */
    public int[] twoSum(int[] numbers, int target) {
        // write your code here
        int[] sum = new int[2];
        for(int i = 0; i<numbers.length; i++) {
            for(int j = i+1; j<numbers.length; j++) {
                if (target - numbers[i] == numbers[j]) {
                    sum[0] = i;
                    sum[1] = j;
                }
            }
        }
        return sum;
    }

    public void swap(int a, int b) {
        int temp;
        temp = a;
        a = b;
        b = temp;
    }

    @Test
    public void test() {
        int[] a = {2, 7, 11, 15};
        System.out.println(Arrays.toString(twoSum(a,9)));
    }
}