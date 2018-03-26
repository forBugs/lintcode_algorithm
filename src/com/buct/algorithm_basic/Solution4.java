package com.buct.algorithm_basic;

public class Solution4 {
    /*
     * @param n: an integer
     * @return: an ineger f(n)
     */
    public int fibonacci(int n) {
        // write your code here
        int a = 0, b = 1;
        if (n == 1)
            return 0;
        if (n == 2) {
            return 1;
        }
        int result = 0;
        for (int i = 3; i <= n; i++) {
            result = a + b;
            a = b;
            b = result;
        }
        return result;


    }

    public int fibonacci2(int n) {
        if (n == 1) {
            return 0;
        }
        if (n == 2) {
            return 1;
        }

        int[] L = new int[n];
        L[0] = 0;
        L[1] = 1;

        for (int i = 2; i < L.length; i++) {
            L[i] = L[i-1] + L[i-2];
        }
        return L[n-1];
    }
}