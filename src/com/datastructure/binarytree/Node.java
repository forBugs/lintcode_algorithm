package com.datastructure.binarytree;

/**
 * 二叉树的节点类型
 * Created by Administrator on 2018/3/18.
 */
public class Node {
    // 数据项
    public long data ;
    public String sData;
    public Node leftChild;
    public Node rightChild;

    public Node(long data, String sData) {
        this.data = data;
        this.sData = sData;
    }

    /**
     * 构造方法
     * @param value
     */
    public Node(long data) {
        this.data = data;
    }
}
