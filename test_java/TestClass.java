package com.example;

import java.util.List;
import java.util.ArrayList;

/**
 * Test class for semantic analysis
 */
public class TestClass {
    private int field1;
    public String field2;
    private List<String> field3;
    
    public TestClass() {
        this.field1 = 0;
        this.field2 = "default";
        this.field3 = new ArrayList<>();
    }
    
    public int getField1() {
        return field1;
    }
    
    public void setField1(int value) {
        if (value > 0) {
            this.field1 = value;
        }
    }
    
    public void complexMethod() {
        for (int i = 0; i < 10; i++) {
            if (i % 2 == 0) {
                System.out.println("Even: " + i);
            } else {
                System.out.println("Odd: " + i);
            }
        }
        
        try {
            // Some complex logic
            for (String item : field3) {
                if (item != null && !item.isEmpty()) {
                    processItem(item);
                }
            }
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
    
    private void processItem(String item) {
        switch (item.length()) {
            case 0:
                System.out.println("Empty item");
                break;
            case 1:
                System.out.println("Single character: " + item);
                break;
            default:
                System.out.println("Long item: " + item);
                break;
        }
    }
    
    public boolean validateData() {
        return field1 > 0 && field2 != null && !field2.isEmpty();
    }
} 