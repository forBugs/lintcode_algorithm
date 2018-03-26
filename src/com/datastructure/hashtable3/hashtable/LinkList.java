package com.datastructure.hashtable3.hashtable;

import java.util.Hashtable;

/**
 * 链表的结构
 * Created by Administrator on 2018/3/20.
 */
public class LinkList {
    // 头结点
    private Node first;

    public LinkList() {
        this.first = null;
    }

    /**
     * 在头结点之后插入新节点
     * @param info
     */
    public void insertFirst(Info info) {
        Node newNode = new Node(info);
        newNode.next = first;
        first = newNode;
    }

    // 删除一个节点  在头结点之后进行删除
    public Node deleteFirst() {
        Node temp = first.next;
        first = temp.next;
        return temp;
    }

    // 根据key查询指定的节点
    public Node find(String key) {
        Node current = first;
        while (!key.equals(current.info.getKey())) {
            if (current.next == null) {
                return null;
            }
            current = current.next;
        }
        return current;
    }

    /**
     * 根据key删除指定的节点
     * @param key
     * @return
     */
    public Node delete(String key) {
        Node current = first;
        Node previous = current;
        // 先查找
        while (!key.equals(current.info.getKey())) {
            if(current.next == null) {
                return null;
            }
            previous = current;
            current = current.next;
        }
        // 再删除
        if (current == first) {
            first = first.next;
        } else {
            previous.next = current.next;
        }
        return current;
    }

}
