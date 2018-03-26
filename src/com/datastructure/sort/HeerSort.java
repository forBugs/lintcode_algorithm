package com.datastructure.sort;

import org.junit.Test;

import java.util.Arrays;

/**
 * 希尔排序 缩小增量排序
 * Created by Administrator on 2018/3/20.
 */
//不稳定
public class HeerSort {

    public  int[] heerSort(int[] a) {

        //希尔排序
        System.out.println();
        int d = a.length/2;
        while(true){
            for(int i=0;i<d;i++){
                for(int j=i;j+d<a.length;j+=d){
                    int temp;
                    if(a[j]>a[j+d]){
                        temp=a[j];
                        a[j]=a[j+d];
                        a[j+d]=temp;
                    }
                }
            }
            if(d==1){break;}
            d--;
        }
        return a;
    }
    //希尔排序
    @Test
    public void test() {
        int[] a={49,38,65,97,76,13,27,49,78,34,12,64,1,33,85,29};

        System.out.println();
        System.out.println("排序之后：");
        System.out.println(Arrays.toString(heerSort(a)));
    }

}