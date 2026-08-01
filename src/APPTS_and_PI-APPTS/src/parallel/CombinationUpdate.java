package parallel;

import engine.Main;

import java.io.Serializable;

public class CombinationUpdate implements Serializable {
    private static final long serialVersionUID = 1L;

    int row;

    int column;

    int newValue;

    private final int[] tuple;

    private final boolean added;

    public CombinationUpdate(int row, int[] tuple) {
        this.row = row;
        this.tuple = tuple;
        this.added = false;
    }

    public CombinationUpdate(int row, int column, int newValue, int[] tuple) {
        this.row = row;
        this.column = column;
        this.newValue = newValue;
        this.tuple = tuple;
        this.added = true;
    }

    // The same task represents removal of the old tuple or addition of the tuple
    // produced by the proposed cell value.
    public void update() {

        int[] value_tuple = new int[Main.t_way];
        if (added) {

            for (int i = 0; i < Main.t_way; i++) {
                if (tuple[i] == column)
                    value_tuple[i] = newValue;
                else
                    value_tuple[i] = Main.coverageArray[row][tuple[i]];
            }
        } else {

            for (int i = 0; i < Main.t_way; i++)
                value_tuple[i] = Main.coverageArray[row][tuple[i]];
        }

        int combinationRow = Main.paraCombination.getRowNum(tuple);

        int combinationColumn = Main.paraValueCombination.getColumnNum(tuple, value_tuple);
        if (added)

            Main.coveredCombination.inc(combinationRow, combinationColumn);
        else

            Main.coveredCombination.dec(combinationRow, combinationColumn);
    }

}
