package com.buct.algorithm_simple;

import sun.text.resources.ar.FormatData_ar;

import javax.naming.directory.SearchResult;
import java.util.PrimitiveIterator;

/**
 * 写出一个高效的算法来搜索 m × n矩阵中的值。

 这个矩阵具有以下特性：

 每行中的整数从左到右是排序的。
 每行的第一个数大于上一行的最后一个整数。
 */
public class Solution28 {
    /*
     * @param matrix: matrix, a list of lists of integers
     * @param target: An integer
     * @return: a boolean, indicate whether matrix contains target
     */
    public static boolean searchMatrix(int[][] matrix, int target) {
        // write your code here
        if (matrix.length==0) return false;
        if (matrix[0].length==0) return false;
        int[] raw_0s = new int[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            raw_0s[i] = matrix[i][0];
            
        }
        int i = 0;
        for (; i < raw_0s.length; i++) {
            if(raw_0s[i] == target) {
                return true;
            }else if (raw_0s[i] > target){
                break;
            }
        }
        System.out.println(i);

        for (int j = 0; j < matrix[0].length; j++) {
            int m = 0;
            if (i!=0) m = i-1;
            if (matrix[m][j]==target) return true;
        }
        return false;
    }

    public static void main(String[] args) {
        int[][] m = {
                {1, 3, 5, 7},
                {10, 11, 16, 20},
                {23, 30, 34, 50}
        };
        int[][] n={{}};
        System.out.println(searchMatrix(n,3));
    }
}