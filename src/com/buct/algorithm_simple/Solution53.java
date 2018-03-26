package com.buct.algorithm_simple;

import java.util.Arrays;

/**
 * 给定一个字符串，逐个翻转字符串中的每个单词。
 */
public class Solution53 {
    /*
     * @param s: A string
     * @return: A string
     */
    public static String reverseWords(String s) {
        // write your code here;
        if (s == null || s == "") {
            return "";
        }
        String[] strs = s.split(" ");
        StringBuilder sb = new StringBuilder();
        for (int i = strs.length-1; i >=0; i--) {
            if (!strs[i].equals(""))
            sb.append(strs[i]).append(" ");
        }
        //去掉最后的空格

        return sb.length()==0?"":sb.substring(0,sb.length()-1);

    }

    public static void main(String[] args) {
        String s = "How are you";
        System.out.println(reverseWords(s));
    }
}