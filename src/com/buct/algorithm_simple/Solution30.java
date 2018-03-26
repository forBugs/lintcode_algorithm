package com.buct.algorithm_simple;
/**
 * 给出一个无重叠的按照区间起始端点排序的区间列表。

 在列表中插入一个新的区间，你要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。
 */

import java.util.ArrayList;
import java.util.List;

/**
 * Definition of Interval:
 *
 */
class Interval{
        int start,end;
        Interval(int start,int end){
        this.start=start;
        this.end=end;
        }
}



public class Solution30 {
    /*
     * @param intervals: Sorted interval list.
     * @param newInterval: new interval.
     * @return: A new interval list.
     */
    public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
        // write your code here
        List<Interval> result = new ArrayList<>();
        int insertIndex = 0;
        for (int i = 0; i < intervals.size(); i++) {
            if (newInterval.start > intervals.get(i).end) {
                result.add(intervals.get(i));
                insertIndex++;
            } else if (newInterval.end < intervals.get(i).start) {
                result.add(intervals.get(i));
            } else {
                newInterval.start = Math.min(newInterval.start, intervals.get(i).start);
                newInterval.end = Math.max(newInterval.end, intervals.get(i).end);
            }
        }


        result.add(insertIndex,newInterval);
        return result;
    }
}