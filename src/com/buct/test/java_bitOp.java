package com.buct.test;

import javax.sound.midi.Soundbank;

/**
 * Created by Administrator on 2017/12/3.
 */
public class java_bitOp {
    public static void main(String[] args) {
        //位运算
        System.out.println(1<<1); //0001 0010
        System.out.println(1<<2); // 0100
        System.out.println(25<<2);

        System.out.println(2>>1);  //0010     0001
        System.out.println(-8>>2);  //0100 0001    10001000 11111000  11111110  100000010

        int a = -2;                            //             11111000
        System.out.println(Integer.toBinaryString(-8));
        System.out.println("---------------------------------");
        //    0000 0101    1111 1011        1111 1111
        System.out.println(5&3);
        System.out.println(-5>>>3);
        System.out.println(5|3);
        System.out.println(5^3);  //抑或运算
        System.out.println(~5);  //~取反运算
        /**
         * 0000 0101  1111 1010 00000101
         * 0000 0011
         * 0000 0111
         * 0000 0001
         * 0000 0110
         */
    }
}
