package com.datastructure.binarytree;

/**
 * Created by Administrator on 2018/3/18.
 */
public class Test {
    @org.junit.Test
    public void test() {
        Tree tree = new Tree();
        tree.insert(10,"A");
        tree.insert(20,"B");
        tree.insert(15,"C");
        tree.insert(3,"Mary");
//        System.out.println(tree.root.data);
//        System.out.println(tree.root.rightChild.data);
//        System.out.println(tree.root.rightChild.leftChild.data);
//        System.out.println(tree.root.leftChild.data);
//        System.out.println(tree.find(15).sData);
        tree.insert(4,"fb");
        tree.insert(90,"fs");
        tree.insert(17,"haha");
//        tree.insert(70,"huge");

        // 前序
//        tree.preOrder(tree.root);
//        tree.inOrder(tree.root);
//        tree.afterOrder(tree.root);

//        tree.delete(4);
        tree.delete(10);
        tree.inOrder(tree.root);
    }


}
