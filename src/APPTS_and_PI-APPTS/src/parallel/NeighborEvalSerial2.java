package parallel;

import java.io.Serializable;

import engine.Main;
import struct.Element;

public class NeighborEvalSerial2 implements Serializable,
        NeighborEval {
    private static final long serialVersionUID = 1L;

    private final int row;

    private final int column;

    private final int newvalue;

    private final Element[] neighbors;

    private final int no;

    public NeighborEvalSerial2(int row, int column, int newvalue, Element[] neighbors, int no) {
        this.row = row;
        this.column = column;
        this.newvalue = newvalue;
        this.neighbors = neighbors;
        this.no = no;
    }

    public void caculEffcAndTabu() {
        int effect1, effect2;
        int[] checkedArray = new int[2 * Main.paraNum];

        for (int k = 0; k < Main.paraNum; k++) {
            checkedArray[2 * k] = k;
            if (k != column)
                checkedArray[2 * k + 1] = Main.coverageArray[row][k];
            else
                checkedArray[2 * k + 1] = newvalue;
        }

        if (Main.numMFTParam[column] > 0)
            effect1 = Main.checker.getMFTNumbyParam(checkedArray, column, newvalue);
        else
            effect1 = 0;

        effect2 = Main.evalMove.evaluate_fast(row, column, newvalue);

        boolean istabu = Main.tabuList.isTabu(row, column, newvalue);

        neighbors[no] = new Element(row, column, newvalue, effect1, effect2, istabu);

    }
}
