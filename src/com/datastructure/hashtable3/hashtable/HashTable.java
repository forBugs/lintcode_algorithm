package com.datastructure.hashtable3.hashtable;

import java.math.BigInteger;

/**
 * 哈希表类  通过链地址法解决哈希冲突：不能保证每个元素都能映射到数组的空白单元
 * Created by Administrator on 2018/3/20.
 */
public class HashTable {
    // 基于链表数组来实现
    private LinkList[] arr;

    /**
     * 默认构造方法
     */
    public HashTable() {
        arr = new LinkList[100];
    }

    /**
     * 指定数组初始化大小
     * @param maxSize
     */
    public HashTable(int maxSize) {
        arr = new LinkList[maxSize];
    }

    /**
     * 插入操作
     * @param info
     */
    public void insert(Info info) {
        // 获取哈希码、
        int hashVal = hashCode(info.getKey());
        //判断当前链表是否为空  为空创建新链表
        if(arr[hashVal] == null) {
            arr[hashVal] = new LinkList();
        }
        // 不为空 向头结点之后插入新元素
        arr[hashVal].insertFirst(info);
    }

    // 从哈希表中查找节点
    public Info find(String key) {
        // 计算哈希码
        int hashVal = hashCode(key);
        // 从对应的链表中查询
        if (arr[hashVal] != null) {
            return arr[hashVal].find(key).info;
        }
        return null;
    }

    // 删除
    public Info delete(String key) {
        // 计算哈希码
        int hashVal = hashCode(key);
        // 从对应的链表中删除
        if (arr[hashVal] != null) {
            return arr[hashVal].delete(key).info;
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
