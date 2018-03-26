package com.buct.algorithm_simple;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

//二叉树的前序遍历
public class Solution66 {
    /*
     * @param root: A Tree
     * @return: Preorder in ArrayList which contains node values.
     */
    public List<Integer> preorderTraversal(TreeNode root) {
        // write your code here
        List<Integer> list = new ArrayList<>();
        if (root==null) return list;
        preorder_data(root,list);
        return list;
    }

    private void preorder_data(TreeNode root, List<Integer> list) {
        if (root==null) return;
        list.add(root.val);
        preorder_data(root.left, list);
        preorder_data(root.right, list);

    }


}