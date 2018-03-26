package com.buct.algorithm_basic;

import java.util.Scanner;
import java.util.Stack;

/**
 * Created by Administrator on 2017/9/23.
 * 用java实现括号匹配    ([])  yes    jfkds(   No
 */
public class P1 {
    public static void main(String[] args){
        Scanner scanner = new Scanner(System.in);
        System.out.println("请输入要判断的括号字符串：");
        String str = scanner.next();
        
        Stack<Character> stack = new Stack<>();
        int length = str.length();
        //如果为奇数 输出no
        if(length%2 == 1){
            System.out.println("No");
        } else {
            for(int i = 0; i<str.length(); i++) {
                 //判断栈为空？  是，入栈
                if(stack.empty()) {
                    stack.push(str.charAt(i));
                } else {
                    //不为空，将当前元素与栈顶元素比较，相同的话出栈
                    if( (stack.lastElement()=='[' && str.charAt(i)==']') || (str.charAt(i)==')' && stack.lastElement()=='(')) {
                        stack.pop();
                    } else {
                        stack.push(str.charAt(i));
                    }

                }
            }

            if(stack.empty()) {
                System.out.println("Yes");
            } else {
                System.out.println("No");
            }
        }

    }

    /**
     * 判断字符串括号是否匹配
     * @param s
     * @return
     */
    public static boolean isMatched(String s){


        return true;
    }


}
