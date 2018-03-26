package com.buct.util;



/**
 * Created by Administrator on 2017/7/12.
 */
public class Demo {
    public static void main(String[] args){
        int x = 10,y = 20;
        swap(x,y);
        System.out.println("x:"+x+" y:"+y);
    }
    public static void swap(int a, int b) {

        int temp;
        temp = a;
        a = b;
        b = temp;

    }


    static {

    }
}