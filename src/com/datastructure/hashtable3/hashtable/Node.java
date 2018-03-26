package com.datastructure.hashtable3.hashtable;

/**
 * 链表的节点
 * Created by Administrator on 2018/3/20.
 */
public class Node {
    // 数据域
    public Info info;
    // 指针域
    public Node next;

    public Node(Info info) {
        this.info = info;
    }
}
