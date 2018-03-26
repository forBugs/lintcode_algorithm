package com.datastructure.binarytree;

/**
 * 二叉树结构
 * Created by Administrator on 2018/3/18.
 */
public class Tree {
    // 根节点
    public Node root;

    /**
     * 插入节点
     * @param value
     */
    public void insert(long value,String sData) {
        // 封装新节点
        Node newNode = new Node(value,sData);
        if (root == null) {
            root = newNode;
            return;
        }
        // 定义当前节点
        Node current = root;
        Node parent;
        while (true) {
            // 定义父节点 通过父节点插入子节点
            parent = current;
            // 遍历寻找插入位置
            if(current.data > value) {
                // 转向左子树
                current = current.leftChild;
                if (current == null) {
                    //插入当前节点
                    parent.leftChild = newNode;
                    return;
                }
            } else {
                // 转向右子树
                current = current.rightChild;
                if (current == null) {
                    parent.rightChild = newNode;
                    return;
                }
            }
        }
    }

    /**
     * 查找节点
     * @param value
     */
    public Node find(long value) {
        // 定义当前节点 初始化为root
        Node current = root;
        while (current.data != value) {
            // 比较当前节点值与value大小
            if (current.data > value) {
                current = current.leftChild;
            } else {
                current = current.rightChild;
            }
            if (current == null) {  // 找不到
                return null;
            }
        }
        return current;
    }

    /**
     * 删除节点
     * @param value
     */
    public boolean delete(long value) {
        // 当前节点
        Node current = root;
        // 当前节点父节点
        Node parent = current;
        // 当前删除节点是否为其父节点的左子节点
        boolean isLeftChild;

        // 先查找被删的节点
        while (current.data != value) {
            parent = current;
            // 比较当前节点值与value大小
            if (current.data > value) {
                current = current.leftChild;
            } else {
                current = current.rightChild;
            }
            if (current == null) {  // 找不到
                return false;
            }
        }

        // 判断|当前删除节点是否为其父节点的左子节点
        if (parent.leftChild == current) {
            isLeftChild = true;
        } else {
            isLeftChild = false;
        }

        // 开始进行删除  分为三种情况
        if (current.leftChild == null && current.rightChild == null) {
            // 1:被删除节点为叶子节点
            if (current == root) {
                root = null;
            }
            if (isLeftChild) {
                parent.leftChild = null;
            } else {
                parent.rightChild = null;
            }
        } else if (current.rightChild == null) {
            // 2:被删除节点有一个左子节点
            if (current == root) {
                root = current.leftChild;
            } else if (isLeftChild) {
                parent.leftChild = current.rightChild;
            } else {
                parent.rightChild = current.rightChild;
            }
        } else if (current.leftChild == null){
            // 被删除节点有一个右子节点
            if (current == root) {
                root = current.rightChild;
            } else if (isLeftChild) {
                parent.leftChild = current.rightChild;
            } else {
                parent.rightChild = current.rightChild;
            }
        } else{
            // 3：被删除节点有两个子节点  找到中序后继节点（比被删除节点值大但最接近的节点）  并用中序后继节点替代被删除节点
            Node successor = getSuccessor(current);
            if (current == root) {
                root = successor;
            } else if (isLeftChild) {
                parent.leftChild = successor;
            } else {
                parent.rightChild = successor;
            }
            successor.leftChild = current.leftChild;
        }

        return true;
    }

    /**
     * 查找被删除节点的中序后继节点
     * @param delNode
     * @return
     */
    public Node getSuccessor(Node delNode) {
        Node successor = delNode;
        Node successorParent = delNode;
        Node current = delNode.rightChild;
        while (current != null) {
            successorParent = current;
            successor = current;
            current = current.leftChild;
        }

        if (successor != delNode.rightChild) {
            successorParent.leftChild = successor.rightChild;
            successor.rightChild = delNode.rightChild;
        }
        return successor;
    }

    /**
     * 二叉树前序遍历
     * @param localNode
     */
    public void preOrder(Node localNode) {
        if (localNode != null) {
            System.out.println(localNode.data + "," + localNode.sData);
            preOrder(localNode.leftChild);
            preOrder(localNode.rightChild);
        }
    }

    /**
     * 中序遍历
     * @param localNode
     */
    public void inOrder(Node localNode) {
        if (localNode != null) {
            inOrder(localNode.leftChild);
            System.out.println(localNode.data + "," + localNode.sData);
            inOrder(localNode.rightChild);
        }
    }

    /**
     * 后序遍历
     * @param localNode
     */
    public void afterOrder(Node localNode) {
        if (localNode != null) {
            afterOrder(localNode.leftChild);
            afterOrder(localNode.rightChild);
            System.out.println(localNode.data + "," + localNode.sData);
        }
    }
}
