package parallel;

import java.util.ArrayList;
import java.util.concurrent.ForkJoinPool;

public class Test {
    public static void main(String[] args) {

        ArrayList<Integer> list1 = new ArrayList<>();
        for (int i = 0; i < 10; i++)
            list1.add(i);
        ArrayList<Integer> list2 = new ArrayList<>();
        for (int i = 0; i < 5; i++)
            list2.add(i);

        ArrayList<Integer> list3 = new ArrayList<>();
        for (int i = 0; i < 5; i++)
            list3.add(i);
        ArrayList<Integer> list4 = new ArrayList<>();
        for (int i = 0; i < 10; i++)
            list4.add(i);

        int row = 0, column = 0;
        int oldValue =0, newValue = 0;
        int len = list1.size()+list2.size()+list3.size()+list4.size();

        ForkJoinPool pool = new ForkJoinPool();

    }
}
