package com.buct.algorithm_simple;

/**
 * 计算a+b,不使用加法运算符
 */
public class Solution11 {
    /*
     * @param : An integer
     * @param : An integer
     * @return: The sum of a and b
     */
    public static int aplusb(int a, int b) {
        // write your code here
        if (a == 0) return b;
        if (b == 0) return a;
        int sum, i;
        i = a^b;
        sum = (a&b)<<1;
        return aplusb(sum,i);
    }

    public static int aplusb2(int a, int b) {
        // write your code here
        //使用循环来解决
        int sum = a^b;   //先不考虑进位进行运算  也就是抑或
        int temp = (a&b)<<1;    //接着考虑进位，就是两个数先按位与，在左移1wei
        while (temp>0) {
            a = sum;
            b = temp;
            sum = a^b;
            temp = (a&b)<<1;
        }
        return sum;

    }

    /**
     * 0000 0101
     * 0001 0001
     * 0001 0100
     *        10
     * @param args
     */

    public static void main(String[] args) {
        System.out.println(aplusb2(10,15));

    }
}