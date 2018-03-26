package com.buct.algorithm_basic;


//Definition of TreeNode:
class TreeNode {
    public int val;
      public TreeNode left, right;
      public TreeNode(int val) {
          this.val = val;
          this.left = this.right = null;
      }

    public TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}
public class Solution {
    /*
     * @param root: the root of tree
     * @return: the max node
     */
    public static TreeNode maxNode(TreeNode root) {
        // write your code here
        if(root==null){
            return null;
        }
        TreeNode leftMax = maxNode(root.left);
        TreeNode rightMax = maxNode(root.right);

        if(leftMax!=null){
            if(leftMax.val > root.val){
                root = leftMax;
            }
        }

        if(rightMax!=null){
            if(rightMax.val > root.val){
                root = rightMax;
            }
        }

        return root;
    }

    public static void main(String[] args){
        TreeNode f31 = new TreeNode(0);
        TreeNode f32 = new TreeNode(3);
        TreeNode f33 = new TreeNode(-4);
        TreeNode f34 = new TreeNode(5);

        TreeNode f21 = new TreeNode(-5);
        f21.left = f31;
        f21.right = f32;
        TreeNode f22 = new TreeNode(2);
        f22.left = f33;
        f22.right = f34;

        TreeNode root = new TreeNode(1);
        root.left = f21;
        root.right = f22;

        // 利用递归分别获取左子树和右字数的最大节点
        TreeNode maxNode = maxNode(root);
        System.out.println("最大节点："+maxNode.val);


    }
}