package combination;

import java.util.ArrayList;

import engine.Main;
import struct.CombinationPosition;

// Tracks every valid t-way value combination by coverage count. A value of -1
// marks a combination excluded by the constraint model.
public class CoveredCombination {
    private final int[][] array;
    private final int rowSize;
    private final int[] pValue;
    private final int t_way;

    private final TupleSet unCoveredTuples;

    private final VarTupleSet onceCoveredTuples;

    public CoveredCombination(int rowSize, int[] pValue, int t_way) {
        int i, j;
        this.rowSize = rowSize;
        this.pValue = pValue;
        this.t_way = t_way;

        int[] tuple = new int[t_way];

        array = new int[rowSize][];
        for (i = 0; i < rowSize; i++) {
            Main.paraCombination.gett_tuple(i, tuple);
            int columnSize = 1;
            for (j = 0; j < t_way; j++)
                columnSize *= Main.pValue[tuple[j]];
            array[i] = new int[columnSize];
        }

        unCoveredTuples = new TupleSet(t_way);
        onceCoveredTuples = new VarTupleSet(t_way);
    }

    public void clear() {
        int i, j;

        unCoveredTuples.clear();
        onceCoveredTuples.clear();

        int[] tuple = new int[t_way];
        int[] value_tuple = new int[t_way];
        int[] checkedArray = new int[t_way * 2];

        // Enumerate each parameter combination and classify invalid value tuples
        // before coverage counts are populated from the current array.
        for (i = 0; i < rowSize; i++) {

            Main.paraCombination.gett_tuple(i, tuple);
            for (j = 0; j < t_way; j++) {
                value_tuple[j] = 0;
                checkedArray[2 * j] = tuple[j];
            }

            while (true) {

                for (int m = t_way - 1; m > 0; m--) {
                    if (value_tuple[m] == pValue[tuple[m]]) {
                        value_tuple[m - 1]++;
                        value_tuple[m] = 0;
                    } else
                        break;
                }

                if (value_tuple[0] == pValue[tuple[0]])
                    break;
                int combinationColumn = Main.paraValueCombination.getColumnNum(tuple, value_tuple);

                for (j = 0; j < t_way; j++)
                    checkedArray[2 * j + 1] = value_tuple[j];

                if (Main.checker.isValid(checkedArray)) {
                    array[i][combinationColumn] = 0;
                }

                else
                    array[i][combinationColumn] = -1;

                value_tuple[t_way - 1]++;
            }
        }
    }

    public void init(int rowNum) {
        int[] tuple = new int[t_way];
        int[] value_tuple = new int[t_way];

        clear();

        for (int combinationrow = 0; combinationrow < rowSize; combinationrow++) {

            Main.paraCombination.gett_tuple(combinationrow, tuple);

            for (int k = 0; k < rowNum; k++) {
                for (int i = 0; i < t_way; i++)
                    value_tuple[i] = Main.coverageArray[k][tuple[i]];
                int combinationColumn = Main.paraValueCombination.getColumnNum(tuple, value_tuple);

                if (array[combinationrow][combinationColumn] != -1) {
                    array[combinationrow][combinationColumn] += 1;
                }
            }
        }

        for (int i = 0; i < rowSize; i++) {

            for (int j = 0; j < array[i].length; j++) {

                if (array[i][j] == 0) {
                    unCoveredTuples.push(i, j);
                }

                else if (array[i][j] == 1) {
                    onceCoveredTuples.push(i, j);
                }
            }
        }
    }

    public void printAllUnCoveredTuple() {
        int[] tuple = new int[t_way];
        int[] value_tuple = new int[t_way];

        ArrayList<CombinationPosition> list = unCoveredTuples.getTupleList();
        for (CombinationPosition elem : list) {
            int combinationRow = elem.combinationRow;
            int combinationColumn = elem.combinationColumn;
            Main.paraCombination.gett_tuple(combinationRow, tuple);
            Main.paraValueCombination.gett_tuple(tuple, combinationColumn, value_tuple);
            for (int j = 0; j < t_way; j++)
                System.out.print(tuple[j] + ":" + value_tuple[j] + " ");
            System.out.println();
        }
    }

    public int getUncoveredCount() {
        return unCoveredTuples.getSize();
    }

    public CombinationPosition getUncoveredCombByList(int p) {
        if (p < 0 || p > unCoveredTuples.getSize() - 1) {
            System.out.println("the value of p is unreasonable in the getUncoveredComb!");
            return null;
        }
        return unCoveredTuples.getTupleList().get(p);
    }

    public ArrayList<CombinationPosition> getUncoveredList() {
        return unCoveredTuples.getTupleList();
    }

    public int getOnlyOnceCoveredCount() {
        return onceCoveredTuples.getOnceCoveredCount();
    }

    public int getOnlyOnceCoveredCountByVar(int para, int value) {
        return onceCoveredTuples.getOnceCoveredCountByVar(para, value);
    }

    // Keep the zero-coverage and exactly-once indexes synchronized with each count.
    public void inc(int row, int column) {
        if (array[row][column] == -1)
            return;
        if (array[row][column] == 0) {
            unCoveredTuples.pop(row, column);
            onceCoveredTuples.push(row, column);
        } else if (array[row][column] == 1) {
            onceCoveredTuples.pop(row, column);
        }
        array[row][column] += 1;
    }

    public void dec(int row, int column) {
        if (array[row][column] == -1)
            return;
        if (array[row][column] == 1) {
            unCoveredTuples.push(row, column);
            onceCoveredTuples.pop(row, column);
        } else if (array[row][column] == 2) {
            onceCoveredTuples.push(row, column);
        }
        array[row][column] -= 1;
    }

    public boolean isOnlyOnce(int row, int column) {
        return array[row][column] == 1;
    }

    public boolean isUncovered(int row, int column) {
        return array[row][column] == 0;
    }

    public void update_row(int changeRowNo, int[] newrow) {
        int i;
        int[] tuple = new int[t_way];
        int[] oldvalue_tuple = new int[t_way];
        int oldCombinationColumn;
        int[] newvalue_tuple = new int[t_way];
        int newCombinationColumn;

        for (int combinationrow = 0; combinationrow < rowSize; combinationrow++) {

            Main.paraCombination.gett_tuple(combinationrow, tuple);

            for (i = 0; i < t_way; i++) {
                oldvalue_tuple[i] = Main.coverageArray[changeRowNo][tuple[i]];
                newvalue_tuple[i] = newrow[tuple[i]];
            }

            oldCombinationColumn = Main.paraValueCombination.getColumnNum(tuple, oldvalue_tuple);
            dec(combinationrow, oldCombinationColumn);

            newCombinationColumn = Main.paraValueCombination.getColumnNum(tuple, newvalue_tuple);
            inc(combinationrow, newCombinationColumn);
        }
    }

    public void add_row(int[] newrow) {
        int i;
        int[] tuple = new int[t_way];
        int[] newvalue_tuple = new int[t_way];
        int newCombinationColumn;

        for (int combinationrow = 0; combinationrow < rowSize; combinationrow++) {

            Main.paraCombination.gett_tuple(combinationrow, tuple);

            for (i = 0; i < t_way; i++) {
                newvalue_tuple[i] = newrow[tuple[i]];
            }

            newCombinationColumn = Main.paraValueCombination.getColumnNum(tuple, newvalue_tuple);
            inc(combinationrow, newCombinationColumn);
        }
    }

    public void delete_row(int deleteRowNo) {
        int[] tuple = new int[t_way];
        int[] value_tuple = new int[t_way];

        for (int combinationrow = 0; combinationrow < rowSize; combinationrow++) {

            Main.paraCombination.gett_tuple(combinationrow, tuple);

            for (int i = 0; i < t_way; i++)
                value_tuple[i] = Main.coverageArray[deleteRowNo][tuple[i]];
            int combinationColumn = Main.paraValueCombination.getColumnNum(tuple, value_tuple);
            dec(combinationrow, combinationColumn);
        }
    }

    public ArrayList<CombinationPosition> getOnceCoveredListByVar(int para, int value) {
        return onceCoveredTuples.getOnceCoveredListByVar(para, value);
    }

}
