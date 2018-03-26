package com.datastructure.hashtable2.hashtable;

import org.junit.Test;

/**
 * Created by Administrator on 2018/3/20.
 */
public class HashTableTest {

    @Test
    public void test() {
        HashTable ht = new HashTable();
        ht.insert(new Info("a", "张三"));
        ht.insert(new Info("ct", "李四"));
        ht.insert(new Info("b", "王五"));
        System.out.println(ht.find("a").getName());
        System.out.println(ht.find("ct").getName());
        System.out.println(ht.find("b").getName());
        ht.delete("a");
        System.out.println(ht.find("a").getName());
    }
}
