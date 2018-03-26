package com.buct.algorithm_simple;

/**
 * 给定一个二叉树，找出所有路径中各节点相加总和等于给定 目标值 的路径。
 * 一个有效的路径，指的是从根节点到叶节点的路径。
 */
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Stack;

/**
 * Definition of TreeNode:
 **/
class TreeNode {
    public int val;
    public TreeNode left, right;

    public TreeNode(int val) {
        this.val = val;
        this.left = this.right = null;
    }
}

public class Solution632 {
    private List<List<Integer>> lists = new ArrayList<List<Integer>>();
    private ArrayList<Integer> list = new ArrayList<Integer>();


    /*
     * @param root: the root of binary tree
     * @param target: An integer
     * @return: all valid paths
     */
    public List<List<Integer>> binaryTreePathSum(TreeNode root, int target) {
        // write your code here
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        if(null==root) return result;
        Stack<Integer> stack = new Stack<>();
        binaryTreePathSum(result,stack,root,0,target);
        return result;
    }

    /**
     * 获取所有的二叉树的路径
     * @param result
     * @param stack
     * @param root
     * @param sum
     * @param target
     */
    public void binaryTreePathSum(List<List<Integer>> result, Stack<Integer> stack, TreeNode root, int sum, int target) {
        sum += root.val;
        stack.push(root.val);
        if (sum == target && root.left == null && root.right == null) {
            List<Integer> list = new ArrayList<>(stack);
            result.add(list);
            stack.pop();
        } else {
            if (root.left != null) {
                binaryTreePathSum(result,stack,root.left,sum,target);
            }
            if (root.right != null) {
                binaryTreePathSum(result,stack,root.right,sum,target);
            }
            stack.pop();
        }
    }







}