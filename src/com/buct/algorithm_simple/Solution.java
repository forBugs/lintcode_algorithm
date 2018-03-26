package com.buct.algorithm_simple;

class Solution {
    /**
     * @return: The same instance of this class every time
     */

    private Solution(){}
    private static final Solution instance = new Solution();
    public static Solution getInstance() {
        // write your code here
        return instance;
    }
};