package com.buct.algorithm_medium;

import org.junit.Test;

import java.util.Stack;

/**
 * 设计一个算法，找出只含素因子2，3，5 的第 n 小的数。 符合条件的数如：1, 2, 3, 4, 5, 6, 8, 9, 10, 12...
 */
public class Solution4 {
    /*
     * @param n: An integer
     * @return: the nth prime number as description.
     */
    public int nthUglyNumber(int n) {
        // write your code here
        if(n==1) return 1;
        int count = 0;
        for(int i=1;;i++){
            if(isUgly(i)) {
                count++;
                if(count == n){
                    return i;
                }
            }
        }
    }

    /**
     * 判断一个数是否为丑数
     * @param num
     * @return
     */
    public boolean isUgly(int num) {
        if (num <= 1) {
            return true;
        }
        Stack<Integer> stack = new Stack<>();
        stack.push(2);
        stack.push(3);
        stack.push(5);
        for (int i = 0; i < stack.size(); i++) {
            //
            while (num % stack.get(i) == 0) {
                num = num / stack.get(i);
            }
        }
        if (num == 1) return true;
        return false;
    }


    @Test
    public void test(){
        System.out.println(nthUglyNumber(1));
    }
}