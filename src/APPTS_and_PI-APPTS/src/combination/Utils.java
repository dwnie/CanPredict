package combination;

import java.util.ArrayList;

public class Utils {

    public static ArrayList<int[]> getParamCombos(int n, int m)
    {

        ArrayList<int[]> res = new ArrayList<>();

        int[] index = new int[n];

        setZeroOrOne(0, n - m, index);

        boolean exhausted = false;

        while (!exhausted) {

            res.add(index.clone());
            int pos = -1;
            int ones = 0;
            int count = 0;

            for (int i = 0; i < n - 1; i++) {
                if (index[i] == 1) {

                    ones++;
                }

                else if (index[i] == 0 && index[i + 1] == 1) {

                    pos = i;

                    count = ones;
                }
            }

            if (pos == -1) {

                exhausted = true;
                continue;
            }

            index[pos] = 1;

            if (index[n - 1] == 1) {

                index[pos + 1] = 0;
            }

            else {

                setZeroOrOne(pos + 1, n - m + count - pos, index);
            }
        }
        return res;
    }

    private static void setZeroOrOne(int start, int count, int[] array)
    {
        for (int i = start; i < array.length; i++) {
            if (i - start < count) {
                array[i] = 0;
            }
            else {
                array[i] = 1;
            }
        }
    }
}
