package com.buct.algorithm_basic;

public class Solution2 {


    //Definition for ListNode.
    public class ListNode {
        int val;
        ListNode next;

        ListNode(int val) {
            this.val = val;
            this.next = null;
        }

        public void print() {

        }
    }


    /*
     * @param head: the first node of linked list.
     * @return: An integer
     */
    public int countNodes(ListNode head) {
        // write your code here
        if (head == null) {
            return 0;
        }

        int count = 0;
        while (head!= null) {
            count++;
            head = head.next;

        }
        return count;
    }

    public static void main(String[] args) {

    }
}