package com.datastructure.hashtable;

import java.math.BigInteger;

/**
 * 哈希表类
 * Created by Administrator on 2018/3/20.
 */
public class HashTable {
    // 基于数组实现
    private Info[] arr;

    /**
     * 默认构造方法
     */
    public HashTable() {
        arr = new Info[100];
    }

    /**
     * 指定数组初始化大小
     * @param maxSize
     */
    public HashTable(int maxSize) {
        arr = new Info[maxSize];
    }

    /**
     * 插入操作
     * @param info
     */
    public void insert(Info info) {
        arr[hashCode(info.getKey())] = info;
    }

    public Info find(String key) {
        return arr[hashCode(key)];

    }

    // 通过String获取哈希值  作为hashtable的key
    public int hashCode(String key) {
//         第一种：通过计算每个字母的ascii码并相加
//        int hashVal= 0;
//        for (int i = key.length()-1; i >= 0; i--) {
//            int letter = key.charAt(i) - 96;
//            hashVal += letter;
//        }
        // 第二种：通过幂的连乘
        BigInteger hashVal = new BigInteger("0");
        BigInteger pow27 = new BigInteger("1");
        for (int i = key.length() - 1; i >=0 ; i--) {
            BigInteger letter = new BigInteger(String.valueOf(key.charAt(i) - 96));
            hashVal = hashVal.add(letter.multiply(pow27));
            pow27 = pow27.multiply(new BigInteger(String.valueOf(27)));
        }
        return hashVal.mod(new BigInteger(String.valueOf(100))).intValue();

    }

}
