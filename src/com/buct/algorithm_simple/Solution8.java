package com.buct.algorithm_simple;

/**
 * 给定一个字符串和一个偏移量，根据偏移量旋转字符串(从左向右旋转)
 */
public class Solution8 {
    /*
     * @param str: An array of char
     * @param offset: An integer
     * @return: nothing
     */
    public static void rotateString(char[] str, int offset) {
        // write your code here
        int length = str.length;
        if (length==0) return;
        int rel_offset = Math.floorMod(offset,length);


        System.out.println(rel_offset);
        char[] temp = new char[rel_offset];
        //
        int index = 0;
        for (int i = length-rel_offset; i < length; i++) {
            temp[index++] = str[i];
        }
        for (int i = length-rel_offset-1; i >=0; i--) {
            str[i+rel_offset] = str[i];
        }

        for (int i = 0; i < rel_offset; i++) {
            str[i] = temp[i];
        }
    }

    public static void main(String[] args) {
        char[] chars = "".toCharArray();
        rotateString(chars,25);
    }
}