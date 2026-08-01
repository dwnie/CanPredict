package ftchecker;

import engine.Main;

import java.util.ArrayList;
import java.util.Iterator;

// Maintains the minimum forbidden tuples currently violated by each covering-array
// row, allowing single-cell moves to update violations incrementally.
public class MFTinCA {

    ArrayList<ArrayList<Tuple>> list_mft_ca;

    int mft_count;

    int rowNum;

    int[] numMFTParam;

    IFTChecker checker;

    public MFTinCA(int rowNum, IFTChecker checker, int[] numMFTParam) {
        mft_count = 0;
        this.checker = checker;
        this.rowNum = rowNum;
        this.numMFTParam = numMFTParam;
        list_mft_ca = new ArrayList<>();
        for (int i = 0; i < rowNum; i++)
            list_mft_ca.add(new ArrayList<>());
    }

    public void clear() {
        for (int i = 0; i < rowNum; i++)
            list_mft_ca.get(i).clear();
        mft_count = 0;
    }

    public int getCount() {
        return mft_count;
    }

    public ArrayList<Tuple> getMFTs(int[] row) {
        int i, j, k = 0;
        ArrayList<Tuple> tuples = new ArrayList<Tuple>();
        for (i = 0; i < list_mft_ca.size(); i++) {
            for (j = 0; j < list_mft_ca.get(i).size(); j++) {
                row[k++] = i;
                tuples.add(list_mft_ca.get(i).get(j));
            }
        }
        return tuples;
    }

    public void deletMFT(int param, int row) {
        int k;

        if (numMFTParam[param] == 0)
            return;

        if(list_mft_ca.get(row).size()==0)
            return;

        int[] rowTuple = new int[Main.paraNum * 2];
        for (int j = 0; j < Main.paraNum; j++) {
            rowTuple[2 * j] = j;
            rowTuple[2 * j + 1] = Main.coverageArray[row][j];
        }

        ArrayList<Tuple> mft_param = checker.getMFTbyParam(rowTuple, param, rowTuple[2 * param + 1]);
        if (mft_param != null)
            for (Iterator<Tuple> iter = mft_param.iterator(); iter.hasNext(); ) {
                Tuple t = iter.next();

                for (k = 0; k < list_mft_ca.get(row).size(); k++)

                    if (t.isEqual(list_mft_ca.get(row).get(k))) {
                        list_mft_ca.get(row).remove(k);
                        mft_count--;
                        break;
                    }
            }
    }

    public void addMFT(int newValue, int param, int row) {

        if (numMFTParam[param] == 0)
            return;

        int[] rowTuple = new int[Main.paraNum * 2];
        for (int j = 0; j < Main.paraNum; j++) {
            rowTuple[2 * j] = j;
            rowTuple[2 * j + 1] = Main.coverageArray[row][j];
        }
        rowTuple[2 *param + 1] = newValue;

        ArrayList<Tuple> mft_param = checker.getMFTbyParam(rowTuple, param, rowTuple[2 * param + 1]);
        if (mft_param != null)
            for (Iterator<Tuple> iter = mft_param.iterator(); iter.hasNext(); ) {
                Tuple t = iter.next();
                list_mft_ca.get(row).add(t);
                mft_count++;
            }
    }

    // Rebuild one row while deduplicating tuples that involve multiple parameters.
    public void updateMFT_row(int rowNo, int[] newRow) {
        int colNum = numMFTParam.length;
        int j, k;
        int[] rowTuple = new int[colNum * 2];

        mft_count -= list_mft_ca.get(rowNo).size();
        list_mft_ca.get(rowNo).clear();

        for (j = 0; j < colNum; j++) {
            rowTuple[2 * j] = j;
            rowTuple[2 * j + 1] = newRow[j];
        }

        for (j = 0; j < colNum; j++) {
            if (numMFTParam[j] == 0)
                continue;

            ArrayList<Tuple> mft_param = checker.getMFTbyParam(rowTuple, j, newRow[j]);
            int mft_row_size = list_mft_ca.get(rowNo).size();

            if (mft_param != null)
                for (Iterator<Tuple> iter = mft_param.iterator(); iter.hasNext(); ) {
                    Tuple t = iter.next();
                    for (k = 0; k < mft_row_size; k++)
                        if (t.isEqual(list_mft_ca.get(rowNo).get(k)))
                            break;

                    if (k == mft_row_size) {
                        list_mft_ca.get(rowNo).add(t);
                        mft_count++;
                    }
                }
        }
    }

    public void addNoMFT_row() {
        int rowNo = list_mft_ca.size();
        list_mft_ca.add(new ArrayList<>());
    }

    public void addMFT_row(int[] newRow) {
        int colNum = numMFTParam.length;
        int j, k;
        int[] rowTuple = new int[colNum * 2];

        int rowNo = list_mft_ca.size();
        list_mft_ca.add(new ArrayList<>());

        for (j = 0; j < colNum; j++) {
            rowTuple[2 * j] = j;
            rowTuple[2 * j + 1] = newRow[j];
        }

        for (j = 0; j < colNum; j++) {
            if (numMFTParam[j] == 0)
                continue;

            ArrayList<Tuple> mft_param = checker.getMFTbyParam(rowTuple, j, newRow[j]);
            int mft_row_size = list_mft_ca.get(rowNo).size();

            if (mft_param != null)
                for (Iterator<Tuple> iter = mft_param.iterator(); iter.hasNext(); ) {
                    Tuple t = iter.next();
                    for (k = 0; k < mft_row_size; k++)
                        if (t.isEqual(list_mft_ca.get(rowNo).get(k)))
                            break;

                    if (k == mft_row_size) {
                        list_mft_ca.get(rowNo).add(t);
                        mft_count++;
                    }
                }
        }
    }
}
