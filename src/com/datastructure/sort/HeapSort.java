package com.datastructure.sort;

import org.junit.Test;

import java.util.Arrays;

/**
 * 堆排序  时间复杂度：n*logn
 * Created by Administrator on 2018/3/20.
 */
public class HeapSort {
    @Test
    public void test() {
        int[] array = {39, 44, 1, 0, 8, 66, 23, 67, 9, 15, 100, 70, 22, 3, 6, 54};
        heapSort(array);
        System.out.println(Arrays.toString(array));
    }

    public void heapSort(int[] a){
        if (a == null || a.length <= 1) {
            return ;
        }
        // 构建初始大堆
        buildMaxHeap(a);
        for (int i = a.length -1; i >=1; i--) {
            // 最大元素已经排在了下标为0的地方  即堆顶  交换元素
            exchangeElements(a,0,i);
            // 对剩下的元素 继续进行大堆构造
            maxHeap(a,i,0);
        }

    }

    // 构造初始最大堆
    private void buildMaxHeap(int[] a) {
        int half = (a.length - 1)/2;  //假设为9
        for (int i = half; i >= 0 ; i--) {
            // 只需遍历 4 3 2 1 0
            maxHeap(a, a.length, i);
        }
    }

    /**
     * 堆调整
     * @param a
     * @param length 用于构造大堆的数组长度元素数量
     * @param i  从第i个元素开始构造
     */
    private void maxHeap(int[] a, int length, int i) {
        int left = i*2 +1;
        int right = i*2 +2;
        int largest = i;
        // 找到三者中最大的数  并与a[i]交换
        if (left < length && a[left] > a[i]) {
            largest = left;
        }
        if(right<length && a[right] > a[largest]){
            largest = right;
        }
        if(i != largest) {
            // 进行数据交换
            exchangeElements(a,i,largest);
            // 继续向下进行大堆构造
            maxHeap(a,length,largest);
        }
    }

    private void exchangeElements(int[] a,int i,int j){
        int temp = a[i];
        a[i] = a[j];
        a[j] = temp;
    }
}



