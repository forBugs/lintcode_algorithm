package com.buct.algorithm_simple;


import java.util.ArrayList;
import java.util.List;

//二叉树的后序遍历
public class Solution68 {
    /*
     * @param root: A Tree
     * @return: Inorder in ArrayList which contains node values.
     */
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        if (root==null) return list;
        postorder_data(root, list);
        return list;
    }

    private void postorder_data(TreeNode root, List<Integer> list) {
        if (root==null) return;
        postorder_data(root.left,list);
        postorder_data(root.right,list);
        list.add(root.val);
    }
}