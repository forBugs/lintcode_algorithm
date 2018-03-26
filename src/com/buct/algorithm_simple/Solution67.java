package com.buct.algorithm_simple;


import java.util.ArrayList;
import java.util.List;
//二叉树的中序遍历
public class Solution67 {
    /*
     * @param root: A Tree
     * @return: Inorder in ArrayList which contains node values.
     */
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        if (root==null) return list;
        inorder_data(root, list);
        return list;
    }

    private void inorder_data(TreeNode root, List<Integer> list) {
        if (root==null) return;
        inorder_data(root.left,list);
        list.add(root.val);
        inorder_data(root.right,list);
    }
}