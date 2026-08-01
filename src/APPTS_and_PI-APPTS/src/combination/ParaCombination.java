package combination;

import engine.Main;

import java.util.ArrayList;

public class ParaCombination {

    private final int[][] tuple_array;
    private final int t_way;

    public ParaCombination(int rowSize, int t_way) {
        this.t_way = t_way;

        tuple_array = new int[rowSize][t_way];

        ArrayList<int[]> list = Utils.getParamCombos(Main.paraNum, t_way);

        for(int k=0,index=list.size()-1; k<list.size(); k++,index--){
            int [] a = list.get(index);
            int count = 0;
            for(int i=0; i<a.length; i++){

                if(a[i]==1){
                    tuple_array[k][count++] = i;
                }
            }
        }
    }

    public int getRowNum(int[] tuple) {
        int sum = 0;
        int deno = factorial(this.t_way-1);
        for (int i = 0; i < this.t_way; i++) {
            int st = i >= 1 ? tuple[i - 1] + 1 : 0;
            int ed = tuple[i];
            int sum1 = 0;
            for (int j = st; j < ed; j++) {
                sum1 += mFromN(Main.paraNum - j - 1, this.t_way - i - 1);
            }
            sum1 /= deno;
            sum += sum1;
            if(i < this.t_way-1)
                deno = deno / (this.t_way - i -1);
        }
        return sum;
    }

    public void gett_tuple(int row, int[] tuple) {
        int i;
        for (i = 0; i < t_way; i++)
            tuple[i] = tuple_array[row][i];
    }

    private static int mFromN(int n, int m) {
        if (m == 0)
            return 1;
        int res = 1;
        int i;
        for (i = n; i>n-m; i--) {
            res *= i;
        }
        return res;
    }

    private static int factorial(int t){
        int res =1;
       for(int i=1; i<=t; i++)
           res *= i;
       return res;
    }

}
