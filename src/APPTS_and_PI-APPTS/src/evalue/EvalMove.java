package evalue;

import java.util.ArrayList;

import combination.Utils;
import engine.Main;
import struct.CombinationPosition;

// Precomputes all t-way parameter combinations containing each column so a move
// can update only the affected coverage counters.
public class EvalMove {

    private int[][] tuple_array;

    private final int[][][] tuple_arrays;

    private final int subRowSize;

    private final int t_way;

    public EvalMove(int t_way) {
        int i;
        this.t_way = t_way;

        int num = 1;
        int denum = 1;
        int para = Main.paraNum - 1;
        int way = t_way - 1;
        for (i = 0; i < t_way - 1; i++) {
            num *= (para--);
            denum *= (way--);
        }
        subRowSize = num / denum;

        tuple_array = new int[subRowSize][t_way];
        tuple_arrays = new int[Main.paraNum][subRowSize][t_way];

        caculArrays();

    }

    private void caculArrays() {

        ArrayList<int[]> list = Utils.getParamCombos(Main.paraNum - 1, t_way - 1);

        for (int k = 0, index = list.size() - 1; k < list.size(); k++, index--) {
            int[] a = list.get(index);
            int count = 0;
            for (int i = 0; i < a.length; i++) {

                if (a[i] == 1) {
                    tuple_array[k][count++] = i;
                }
            }
        }

        for (int k = 0; k < Main.paraNum; k++) {
            for (int r = 0; r < tuple_array.length; r++) {
                int i;

                for (i = 0; i < t_way - 1 && tuple_array[r][i] < k; i++)
                    tuple_arrays[k][r][i] = tuple_array[r][i];

                tuple_arrays[k][r][i] = k;

                for (int j = i; j < t_way - 1; j++)
                    tuple_arrays[k][r][j + 1] = tuple_array[r][j] + 1;
            }
        }
    }

    public  int[][] getParaCombsByPara(int column) {
       return tuple_arrays[column];
    }

    // Negative effects cover previously uncovered tuples; positive effects remove
    // the only remaining coverage of a tuple.
    public int evaluate_fast(int row, int column, int newValue) {
        int effect = 0;
        int[] tuple = new int[t_way];
        int[] value_tuple = new int[t_way];

        ArrayList<CombinationPosition> unCoveredList = Main.coveredCombination.getUncoveredList();

        for (CombinationPosition element : unCoveredList) {

            int combinationRow = element.combinationRow;
            Main.paraCombination.gett_tuple(combinationRow, tuple);

            int combinationColumn = element.combinationColumn;
            Main.paraValueCombination.gett_tuple(tuple, combinationColumn, value_tuple);

            boolean flag = true;

            for (int j = 0; j < t_way && flag; j++) {
                if (tuple[j] != column) {
                    if (value_tuple[j] != Main.coverageArray[row][tuple[j]])
                        flag = false;
                } else {
                    if (value_tuple[j] != newValue)
                        flag = false;
                }
            }

            if (flag)
                effect--;
        }

        ArrayList<CombinationPosition> onceCoveredList = Main.coveredCombination.getOnceCoveredListByVar(column,
                Main.coverageArray[row][column]);

        for (CombinationPosition element : onceCoveredList) {

            int combinationRow = element.combinationRow;
            Main.paraCombination.gett_tuple(combinationRow, tuple);

            int combinationColumn = element.combinationColumn;
            Main.paraValueCombination.gett_tuple(tuple, combinationColumn, value_tuple);

            boolean flag = true;

            for (int j = 0; j < t_way && flag; j++) {
                if (value_tuple[j] != Main.coverageArray[row][tuple[j]]) {
                    flag = false;
                    break;
                }
            }

            if (flag)
                effect++;
        }

        return effect;
    }

    public void move(int row, int column, int newValue) {

        int i, z;

        int[] oldTupleValue = new int[t_way];
        int[] newTupleValue = new int[t_way];
        int combinationRow = 0, oldCombinationColumn, newCombinationColumn;

        tuple_array = getParaCombsByPara(column);

        for (z = 0; z < subRowSize; z++) {

            for (i = 0; i < t_way; i++) {
                oldTupleValue[i] = Main.coverageArray[row][tuple_array[z][i]];

                if (tuple_array[z][i] == column)
                    newTupleValue[i] = newValue;
                else
                    newTupleValue[i] = oldTupleValue[i];
            }

            combinationRow = Main.paraCombination.getRowNum(tuple_array[z]);

            oldCombinationColumn = Main.paraValueCombination.getColumnNum(tuple_array[z], oldTupleValue);
            newCombinationColumn = Main.paraValueCombination.getColumnNum(tuple_array[z], newTupleValue);

            Main.coveredCombination.dec(combinationRow, oldCombinationColumn);

            Main.coveredCombination.inc(combinationRow, newCombinationColumn);
        }

    }

}
