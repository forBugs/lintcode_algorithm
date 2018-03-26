package com.buct.algorithm_basic;


//  Definition for singly-linked list.

class ListNode {
    int val;
    ListNode next;

    ListNode(int x) {
        val = x;
    }
}
public class Solution3 {

    /*
     * @param head: a ListNode
     * @param val: An integer
     * @return: a ListNode
     */
    public ListNode removeElements(ListNode head, int val) {
        // write your code here

        ListNode temp = head;
        if (head == null) {
            return null;
        }

        while (temp.next != null) {
            // 找到前一个节点和后一个节点
            if (temp.next.val == val) {
                if (temp.next.next != null) {
                    temp.next = temp.next.next;
                } else {
                    temp.next = null;
                }
            } else {
                temp = temp.next;
            }
        }

        if(head.val==val) {
            return head.next;
        }
        return head;
    }

//    给出链表 1->2->3->3->4->5->3, 和 val = 3, 你需要返回删除3之后的链表：1->2->4->5。
    public static void main(String[] args) {

    }
}



