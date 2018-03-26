package com.datastructure.hashtable2.hashtable;

import java.math.BigInteger;

/**
 * 哈希表类  通过空闲地址法解决哈希冲突：不能保证每个元素都能映射到数组的空白单元
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
        // 获取哈希码、
        int hashVal = hashCode(info.getKey());
        // 遍历数组 找到一个空白的插入位置 如果这个索引已经被引用 而且里面是一个未被删除的数据
        while (arr[hashVal] != null && arr[hashVal].getName() != null) {
            // 递加
            ++hashVal;
            // 循环
            hashVal %= arr.length;
        }
        arr[hashVal] = info;
    }


    public Info find(String key) {
        // 计算哈希码
        int hashVal = hashCode(key);
        // 遍历  计算hashVal  比较当前的key与被查找的key的内容是否相同
        while (arr[hashVal] != null) {
            if (arr[hashVal].getKey().equals(key)) {
                return arr[hashVal];
            }
            ++hashVal;
            hashVal %= arr.length;
        }
        return null;

    }

    // 删除
    public Info delete(String key) {
        // 计算哈希码
        int hashVal = hashCode(key);
        // 遍历  查找到 并删除之
        while (arr[hashVal] != null) {
            if (arr[hashVal].getKey().equals(key)) {
                Info temp = arr[hashVal];
                temp.setName(null);
//                temp.setKey(null);
                return temp;
            }
            ++hashVal;
            hashVal %= arr.length;
        }
        return null;
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
