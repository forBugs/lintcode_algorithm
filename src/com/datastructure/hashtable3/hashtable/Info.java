package com.datastructure.hashtable3.hashtable;

/**
 * 员工信息类
 * Created by Administrator on 2018/3/20.
 */
public class Info {
    private String key;  //  员工信息对应的键
    private String name;

    public Info(String key, String name) {
        this.key = key;
        this.name = name;
    }

    public String getKey() {
        return key;
    }

    public void setKey(String key) {
        this.key = key;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
}
