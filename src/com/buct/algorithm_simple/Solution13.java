package com.buct.algorithm_simple;

import java.util.PrimitiveIterator;

/**
 * 对于一个给定的 source 字符串和一个 target 字符串，你应该在 source 字符串中找出 target 字符串出现的第一个位置(从0开始)。如果不存在，则返回 -1。
 */
public class Solution13 {
    /*
     * @param source: source string to be scanned.
     * @param target: target string containing the sequence of characters to match
     * @return: a index to the first occurrence of target in source, or -1  if target is not part of source.
     */
    public static int strStr(String source, String target) {
        // write your code here
        if (source==null) return -1;
        if (target==null && source!=null) return -1;
        int s_length = source.length();
        int t_length = target.length();
        if (t_length>s_length) return -1;
        for (int i = 0; i <= s_length-t_length; i++) {
            System.out.println("  i:"+i);
            int j = 0;
            for (; j < t_length; j++) {
                System.out.println("j:" + j);
                if(target.charAt(j) != source.charAt(i+j)){
                    break;
                }
            }
            System.out.println("j__:"+j);
            if (j==t_length)
                return i;
        }

        return -1;
    }

    public static void main(String[] args) {
        String source = "abcdabcdefg";
        String target = "bcd";
        System.out.println(strStr(source,target));

    }
}